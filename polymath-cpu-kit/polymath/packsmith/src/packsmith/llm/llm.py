"""OpenAI-compatible LLM client (standard library only): LM Studio, Ollama, OpenRouter, vLLM, llama.cpp server, OpenAI
or any server with /v1/chat/completions and /v1/models (Settings → LLM).

The core uses only the standard API. Model management (list loaded models with their context, load, unload: used to
share one GPU with the engine and to free it for training) is an extra that LM Studio and Ollama provide; with other
servers those calls do nothing."""
from __future__ import annotations

import base64
import json
import re
import time
import http.client
import shutil
import subprocess
import urllib.parse

from packsmith.system.config import SETTINGS


class LLMError(RuntimeError):
    pass


class Cancelled(LLMError):
    """The job this call belongs to was cancelled or paused."""


class EmptyReply(LLMError):
    """The model answered with no text (thinking ran out of budget, or the prompt was cut)."""


def preset() -> str:
    return (SETTINGS.llm_preset or "custom").strip().lower()


def server_name() -> str:
    from packsmith.system.config import PRESETS
    return PRESETS.get(preset(), ("", "the LLM server"))[1].split(" (")[0]


def in_container() -> bool:
    import os
    return os.path.exists("/.dockerenv") or bool(os.environ.get("POLYMATH_IN_DOCKER"))


def normalise_url(url: str) -> str:
    """The address as typed, made usable: 'localhost:1234' -> 'http://localhost:1234/v1'. Without a scheme the
    standard library reads the host as the scheme and connects to port 80, which shows up as "connection refused"."""
    u = (url or "").strip().rstrip("/")
    if not u:
        return u
    if "://" not in u:
        u = "http://" + u
    sp = urllib.parse.urlsplit(u)
    if sp.path in ("", "/"):
        u = urllib.parse.urlunsplit((sp.scheme, sp.netloc, "/v1", sp.query, ""))
    return u


def server_url() -> str:
    """The address requests go to. The config is shared between a direct install and Docker, where "this machine" has
    different names: inside a container localhost is the container itself, so a local address is sent to the host
    (host.docker.internal); outside, host.docker.internal usually does not resolve, so it is sent to 127.0.0.1."""
    u = normalise_url(SETTINGS.llm_url)
    sp = urllib.parse.urlsplit(u)
    host = (sp.hostname or "").lower()
    swap = ""
    if in_container() and host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        swap = "host.docker.internal"
    elif not in_container() and host == "host.docker.internal":
        import socket
        try:
            socket.gethostbyname(host)
        except OSError:
            swap = "127.0.0.1"
    if swap:
        u = urllib.parse.urlunsplit((sp.scheme, swap + (f":{sp.port}" if sp.port else ""), sp.path, sp.query, ""))
    return u


def refused_hint(url: str) -> str:
    """What to check when the server does not accept the connection."""
    sp = urllib.parse.urlsplit(url)
    where = f"{sp.hostname}:{sp.port or 80}"
    name = server_name()
    if preset() == "lmstudio":
        start = "start its server (LM Studio → Developer → Start server, or `lms server start`) and check the port (1234)"
        lan = "LM Studio → Developer → Server Settings → Serve on Local Network"
    elif preset() == "ollama":
        start, lan = "start it (`ollama serve`) and check the port (11434)", "start Ollama with OLLAMA_HOST=0.0.0.0"
    else:
        start, lan = "start the server and check the port", "make the server listen on 0.0.0.0, not only 127.0.0.1"
    if in_container():
        return (f"nothing accepts connections at {where}. Packsmith runs in Docker, so {name} on your machine is reached "
                f"through host.docker.internal and must listen on the network, not only on localhost: {lan}. Also {start}.")
    if (sp.hostname or "") in ("localhost", "127.0.0.1", "::1"):
        return f"nothing accepts connections at {where}: {start}."
    return f"nothing accepts connections at {where}: {start}; on another machine it must also listen on the network ({lan})."


def is_local() -> bool:
    host = (urllib.parse.urlsplit(server_url()).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal", "ollama", "lmstudio") \
        or host.startswith(("192.168.", "10.", "172."))


def can_manage_models() -> bool:
    """Can Packsmith load/unload this server's models (to share a GPU)? LM Studio and Ollama, when local."""
    return preset() in ("lmstudio", "ollama") and is_local()


def _base() -> str:
    """The server root (the OpenAI path /v1 removed)."""
    return server_url().rstrip("/").rsplit("/v1", 1)[0]


def _headers() -> dict:
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {SETTINGS.api_key() or 'no-key'}"}
    if preset() == "openrouter":
        h.update({"HTTP-Referer": "https://github.com/polymath", "X-Title": "Packsmith"})
    return h


def _http(url: str, body: dict | None = None, timeout: float = 600) -> dict:
    """One JSON request over http.client. The connection is registered with the running job, so cancelling the job
    closes it at once (a long generation stops instead of running on to its timeout)."""
    from packsmith.system import jobs
    u = urllib.parse.urlsplit(url)
    conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout)
    job = jobs.current()
    if job is not None:
        if job.cancelled():
            raise Cancelled("cancelled")
        job.conns.add(conn)
    try:
        data = json.dumps(body).encode() if body is not None else None
        conn.request("POST" if body is not None else "GET", (u.path or "/") + (f"?{u.query}" if u.query else ""), body=data,
                     headers=_headers())
        r = conn.getresponse()
        raw = r.read()
        if r.status >= 400:
            raise LLMError(f"{server_name()} {r.status}: {raw[:400].decode(errors='replace')}")
        return json.loads(raw or b"{}")
    except (OSError, http.client.HTTPException) as e:
        if job is not None and job.cancelled():
            raise Cancelled("cancelled") from e
        if isinstance(e, ConnectionRefusedError):
            raise LLMError(f"{server_name()} not reachable at {url}: {refused_hint(url)}") from e
        raise LLMError(f"{server_name()} not reachable at {url}: {e}") from e
    finally:
        if job is not None:
            job.conns.discard(conn)
        conn.close()


def _req(path: str, body: dict | None = None, timeout: float = 600) -> dict:
    return _http(server_url().rstrip("/") + path, body, timeout)


def instances() -> list[dict]:
    """The server's models with their loaded instances: [{key, type, loaded: [{id, context_length, parallel}]}].
    LM Studio: GET /api/v1/models. Ollama: GET /api/ps (loaded models only). Empty for other servers."""
    if not can_manage_models():
        return []
    if preset() == "ollama":
        try:
            r = _http(_base() + "/api/ps", timeout=10)
        except LLMError:
            return []
        return [{"key": m.get("model") or m.get("name"), "type": "llm",
                 "loaded": [{"id": m.get("model") or m.get("name"), "context_length": m.get("context_length"), "parallel": None}]}
                for m in r.get("models") or []]
    try:
        r = _http(_base() + "/api/v1/models", timeout=10)
    except LLMError:
        return []
    out = []
    for m in r.get("models", r.get("data", [])) or []:
        li = [{"id": i.get("id"), "context_length": (i.get("config") or {}).get("context_length"),
               "parallel": (i.get("config") or {}).get("parallel")} for i in m.get("loaded_instances") or []]
        out.append({"key": m.get("key") or m.get("id"), "type": m.get("type", ""), "loaded": li})
    return out


def unload_all() -> list[str]:
    """Unload every loaded model (LM Studio: REST API, the lms CLI as a fallback; Ollama: keep_alive 0).
    Returns what was unloaded; nothing for servers that cannot."""
    done = []
    if not can_manage_models():
        return done
    for m in instances():
        for i in m["loaded"]:
            try:
                if preset() == "ollama":
                    _http(_base() + "/api/generate", {"model": i["id"], "keep_alive": 0}, timeout=60)
                else:
                    _http(_base() + "/api/v1/models/unload", {"instance_id": i["id"]}, timeout=60)
                done.append(i["id"])
            except LLMError:
                pass
    if not done and preset() == "lmstudio" and shutil.which("lms"):
        subprocess.run(["lms", "unload", "--all"], capture_output=True, timeout=60)
        done.append("lms unload --all")
    return done


def load(model: str, context_length: int | None = None) -> dict:
    if not can_manage_models():
        return {}
    if preset() == "ollama":
        body = {"model": model, "keep_alive": "30m"}
        if context_length:
            body["options"] = {"num_ctx": int(context_length)}
        return _http(_base() + "/api/generate", body, timeout=600)
    body = {"model": model}
    if context_length:
        body["context_length"] = int(context_length)
    return _http(_base() + "/api/v1/models/load", body, timeout=600)


def models() -> list[dict]:
    """[{id, context, parallel, loaded}] — OpenAI /v1/models, with the loaded context and parallel slots from the
    native API (falls back to /api/v0)."""
    ids = [m["id"] for m in _req("/models", timeout=10).get("data", [])]
    info = {}
    for m in instances():
        if m["loaded"]:
            info[m["key"]] = m["loaded"][0]
            for i in m["loaded"]:
                info[i["id"]] = i
    if not info and preset() == "lmstudio":
        try:
            for m in _http(_base() + "/api/v0/models", timeout=5).get("data", []):
                info[m.get("id")] = {"context_length": m.get("loaded_context_length") or m.get("max_context_length")}
        except LLMError:
            pass
    return [{"id": i, "context": (info.get(i) or {}).get("context_length"), "parallel": (info.get(i) or {}).get("parallel"),
             "loaded": i in info} for i in ids]


_CTX: dict = {"t": 0.0, "map": {}}


def context_tokens(model: str | None = None) -> int | None:
    """The loaded context length of a model (cached 60 s); None when the server does not say."""
    if time.time() - _CTX["t"] > 60 or _CTX.get("url") != SETTINGS.llm_url:     # a new server: ask again
        _CTX["url"] = SETTINGS.llm_url
        try:
            _CTX["map"] = {m["id"]: m["context"] for m in models()}
        except LLMError:
            _CTX["map"] = {}
        _CTX["t"] = time.time()
    try:
        m = model or default_model()
    except LLMError:
        return None
    return _CTX["map"].get(m)


CHARS_PER_TOKEN = 3.5          # English prose; on the safe side for numbers and tables


def data_chars(tokens: int | None = None) -> int:
    """Settings.data_tokens (source text per call) in characters."""
    return int((tokens or SETTINGS.data_tokens) * CHARS_PER_TOKEN)


def budget_chars(model: str | None, out_tokens: int, overhead_chars: int = 3000, cap_tokens: int | None = None) -> int:
    """How much source text fits in one call: Settings.data_tokens (or cap_tokens), and never more than the model's
    loaded context minus the reply and the prompt's own text."""
    cap = data_chars(cap_tokens)
    out_tokens = max(int(out_tokens), int(SETTINGS.reply_tokens or 0))   # the reply (thinking included) needs its room
    ctx = context_tokens(model)
    if not ctx:
        return cap
    fit = int((ctx - out_tokens - 1000) * 3) - overhead_chars
    if fit < 2500:
        raise LLMError(f"the model's loaded context ({ctx} tokens) is too small for this step: load it in the LLM server "
                       f"with a context length of at least {out_tokens + 4000} tokens (8192 or more recommended)")
    return min(cap, fit)


def default_model() -> str:
    if SETTINGS.default_model:
        return SETTINGS.default_model
    ms = models()
    if not ms:
        raise LLMError(f"{server_name()} lists no models: load or pull one first, or set Settings → LLM → Model for mining")
    return ms[0]["id"]


_THINK = re.compile(r"<think>.*?</think>", re.S)


def clean(text: str) -> str:
    text = _THINK.sub("", text or "")
    m = re.search(r"```(?:html)?\s*(.*?)```", text, re.S)
    if m and "<" in m.group(1):
        text = m.group(1)
    return text.strip()


def _gpu() -> None:
    """Shared GPU: make sure the LLM server holds it (stops the engine, loads remembered models)."""
    from packsmith.system import gpu, jobs
    j = jobs.current()
    gpu.need_lmstudio((lambda m: j.report(j.progress, m)) if j else (lambda m: None))


def reply_budget(asked: int, messages: list[dict], model: str | None) -> int:
    """Tokens LM Studio may write for one reply. LM Studio counts a model's thinking inside max_tokens, so every call
    gets at least Settings.reply_tokens (thinking + answer), never more than the loaded context leaves after the
    prompt (about 3 characters per token)."""
    want = max(int(asked or 0), int(SETTINGS.reply_tokens or 0))
    ctx = context_tokens(model)
    if ctx:
        prompt_tok = sum(len(m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))) for m in messages) // 3
        want = min(want, max(1024, ctx - prompt_tok - 256))
    return want


def _body(messages, model, max_tokens, temperature) -> dict:
    m = model or default_model()
    body = {"model": m, "messages": messages, "max_tokens": reply_budget(max_tokens, messages, m), "temperature": temperature,
            "stream": False}
    if SETTINGS.reasoning_effort:            # only where the server takes it (a strict server rejects unknown fields)
        if preset() in ("lmstudio", "openai", "ollama"):
            body["reasoning_effort"] = SETTINGS.reasoning_effort
        elif preset() == "openrouter":
            body["reasoning"] = {"effort": SETTINGS.reasoning_effort}
    return body


def chat(messages: list[dict] | str, model: str | None = None, max_tokens: int = 4096, temperature: float = 0.2,
         timeout: float = 900, retries: int = 2) -> str:
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    _gpu()
    body = _body(messages, model, max_tokens, temperature)
    last = None
    for attempt in range(retries + 1):
        try:
            r = _req("/chat/completions", body, timeout)
            ch = r["choices"][0]
            text = clean(ch["message"].get("content") or "")
            if not text:
                thinking = bool(ch["message"].get("reasoning_content") or ch["message"].get("reasoning")) or ch.get("finish_reason") == "length"
                if thinking and ch.get("finish_reason") == "length":
                    grown = reply_budget(body["max_tokens"] * 2, messages, body["model"])
                    if grown > body["max_tokens"]:           # the thinking ran out of room: try once more with twice the room
                        body["max_tokens"] = grown
                why = (f"it spent the whole reply budget ({body['max_tokens']} tokens, thinking included) thinking and gave no "
                       "answer: raise Settings → Reply budget, set Reasoning effort to low, or load the model with more "
                       "context") if thinking else "it returned nothing"
                raise EmptyReply(f"empty reply from {body['model']}: {why}")
            return text
        except LLMError as e:
            last = e
            if isinstance(e, Cancelled) or (not isinstance(e, EmptyReply) and "not reachable" not in str(e)
                                            and "50" not in str(e)[:16]):
                raise
            time.sleep(2 * (attempt + 1))
    raise last


def vision(image: bytes, prompt: str, model: str | None = None, max_tokens: int = 4096) -> str:
    url = "data:image/png;base64," + base64.b64encode(image).decode()
    msg = [{"role": "user", "content": [{"type": "text", "text": prompt},
                                        {"type": "image_url", "image_url": {"url": url}}]}]
    return chat(msg, model=model or SETTINGS.vision_model or None, max_tokens=max_tokens, temperature=0.0)


def chat_raw(messages: list[dict], tools: list[dict] | None = None, model: str | None = None, max_tokens: int = 2000,
             temperature: float = 0.3, timeout: float = 600) -> dict:
    """One chat turn returning the whole message (content and tool_calls); used by the assistant."""
    _gpu()
    body = _body(messages, model, max_tokens, temperature)
    if tools:
        body["tools"] = tools
    r = _req("/chat/completions", body, timeout)
    msg = r["choices"][0]["message"]
    msg["content"] = _THINK.sub("", msg.get("content") or "").strip()
    return msg
