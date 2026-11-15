"""Settings.

Where things live:
  * POLYMATH_HOME (default ~/.polymath of the REAL user, also under sudo) holds config.toml, data/, models/, runs/,
    engines/ — code stays in the repo, state stays here (a volume in Docker).
  * config.toml is grouped in tables ([llm], [gpu], [engine], ...) whose keys are the field names below; a flat file
    (the old packsmith.toml) loads too, and its old key names are migrated (lmstudio_url -> llm_url, ...).
  * Shared folders can be moved with POLYMATH_HF_CACHE (Hugging Face models), POLYMATH_MODELS, POLYMATH_PACKS,
    POLYMATH_DATA — handy for one model/pack folder used by several installs or containers.
  * Environment variables PACKSMITH_<FIELD> override the file (POLYMATH_CONFIG / PACKSMITH_CONFIG name the file).
  * Paths may start with ~ : it is the real user's home, never /root under sudo.
"""
from __future__ import annotations

import json
import os
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from packsmith.system.home import expand, polymath_home

PACKAGE_DIR = Path(__file__).resolve().parents[1]          # .../src/packsmith
HOME = polymath_home()


SECRETS = ("api_token", "llm_api_key")      # never returned by the API: shown as set / not set


@dataclass
class Settings:
    data_dir: str = "${POLYMATH_HOME}/data"
    # ---- LLM: any OpenAI-compatible server (LM Studio, Ollama, OpenRouter, vLLM, llama.cpp, OpenAI, ...)
    llm_preset: str = "lmstudio"
    llm_url: str = "http://127.0.0.1:1234/v1"
    llm_api_key_env: str = "POLYMATH_LLM_API_KEY"   # the key is read from this environment variable ...
    llm_api_key: str = ""                           # ... or written here (kept in config.toml, mode 600)
    default_model: str = ""             # model id for mining (empty = first listed)
    vision_model: str = ""              # model id for OCR (empty = default_model)
    data_tokens: int = 32000            # source text per LLM call, in tokens (prompt and reply come on top)
    capture_tokens: int = 8000          # source text per data-capture call: capture writes out all it reads, so smaller
    reply_tokens: int = 16384           # least reply budget per call (thinking models count thinking inside it)
    reasoning_effort: str = ""          # "" (model default) | low | medium | high — sent only to servers that take it
    parallel: int = 4                   # LLM calls in flight (match the server's parallel slots)
    llm_unload_for_gpu: bool = True     # unload the LLM's models before training / engine use (local servers that can)
    # ---- GPU
    gpu_engine: str = "auto"            # auto | cpu | <index> | <GPU-uuid>
    gpu_train: str = "same"             # same (as the engine) | auto | <index> | <GPU-uuid>
    gpu_llm: str = "same"               # where the LLM server runs: same (card as the engine) | other | none (remote)
    # ---- Engine
    engine: str = "patched-colibri"     # patched-colibri | polymath-engine | polymath-c
    engine_build: str = "auto"          # auto | gpu | cpu: which colibri build serves (both can be installed)
    pack_budget: int = 2                # Polymath Engine: most pack experts per token in a layer (of the model's top-k)
    colibri_dir: str = "${POLYMATH_HOME}/engines/colibri"
    base_serve_model: str = ""          # engine container of the base model
    base_train_model: str = ""          # container or Hugging Face folder of the same base
    hf_cache_dir: str = "~/.cache/huggingface/hub"
    models_dir: str = "${POLYMATH_HOME}/models"   # Polymath's own downloaded models (searched after the HF cache)
    packs_dir: str = "${POLYMATH_HOME}/packs"     # trained packs: one shared folder (a volume in Docker)
    runs_dir: str = "${POLYMATH_HOME}/runs"   # converted models and research outputs (was moe-knowledge/runs)
    engine_ctx: int = 16384             # colibri tokens per request (prompt + answer); Q36_MAXT
    engine_temperature: float = 0.3
    engine_top_p: float = 0.9
    # ---- Training
    train_batch: int = 2
    train_accum: int = 4                # gradient accumulation: effective batch = batch x accum
    train_epochs: int = 0               # 0 = auto (about 4,800 texts seen in all; 2 to 16 epochs)
    train_max_items: int = 0            # 0 = every item
    # ---- Server
    host: str = "127.0.0.1"
    port: int = 8770
    api_token: str = ""
    agent_folders: str = ""             # folders agents may name as user-pack sources (comma-separated); empty = none

    def path(self, key: str) -> Path:  # noqa: D401
        """The setting as a path (~ = the real user's home). An empty setting gives an empty-named path that exists
        nowhere (never the current folder)."""
        v = str(getattr(self, key) or "")
        return expand(v) if v else Path("/nonexistent/unset-" + key)

    def api_key(self) -> str:
        return os.environ.get(self.llm_api_key_env or "", "") or self.llm_api_key or ""


SECTIONS = {
    "llm": ["llm_preset", "llm_url", "llm_api_key_env", "llm_api_key", "default_model", "vision_model", "data_tokens",
            "capture_tokens", "reply_tokens", "reasoning_effort", "parallel", "llm_unload_for_gpu"],
    "gpu": ["gpu_engine", "gpu_train", "gpu_llm"],
    "engine": ["engine", "engine_build", "pack_budget", "colibri_dir", "base_serve_model", "base_train_model", "engine_ctx", "engine_temperature",
               "engine_top_p"],
    "paths": ["data_dir", "packs_dir", "runs_dir", "hf_cache_dir", "models_dir"],
    "training": ["train_batch", "train_accum", "train_epochs", "train_max_items"],
    "server": ["host", "port", "api_token", "agent_folders"],
}

# old key -> (new key, value converter)
LEGACY = {
    "lmstudio_url": ("llm_url", str),
    "lmstudio_key": ("llm_api_key", lambda v: "" if v == "lm-studio" else str(v)),
    "unload_lmstudio": ("llm_unload_for_gpu", lambda v: v),
    "engine_gpu": ("gpu_engine", lambda v: "auto" if str(v).lower() in ("1", "true", "yes", "on") else "cpu"),
    "shared_gpu": ("gpu_llm", lambda v: "same" if str(v).lower() in ("1", "true", "yes", "on") else "other"),
    "moe_knowledge_dir": ("runs_dir", lambda v: str(Path(str(v)) / "runs")),
}

PRESETS = {      # preset -> (default URL, help)
    "lmstudio": ("http://127.0.0.1:1234/v1", "LM Studio (Developer → Start server)"),
    "ollama": ("http://127.0.0.1:11434/v1", "Ollama"),
    "openrouter": ("https://openrouter.ai/api/v1", "OpenRouter (set an API key)"),
    "vllm": ("http://127.0.0.1:8000/v1", "vLLM"),
    "llamacpp": ("http://127.0.0.1:8080/v1", "llama.cpp server"),
    "openai": ("https://api.openai.com/v1", "OpenAI (set an API key)"),
    "custom": ("", "Any other OpenAI-compatible server"),
}


def _coerce(f, v):
    t = f.type if isinstance(f.type, str) else f.type.__name__
    if t == "bool":
        return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
    if t == "int":
        return int(float(v))
    if t == "float":
        return float(v)
    return str(v)


# what the Settings tab shows: group, label, help, input kind, restart needed[, options]
HELP = {
    "llm_preset": ("LLM", "Provider", "Which OpenAI-compatible server mines, generates and judges. Picking one fills in its usual "
                   "address; LM Studio and Ollama can also load/unload models to share a GPU.", "choice", False, list(PRESETS)),
    "llm_url": ("LLM", "Server URL", "The OpenAI-compatible address, ending in /v1.", "text", False),
    "llm_api_key_env": ("LLM", "API key variable", "Environment variable holding the API key (OpenRouter, OpenAI). "
                        "Preferred over writing the key below.", "text", False),
    "llm_api_key": ("LLM", "API key", "Only if you cannot use the variable above. Saved in config.toml (readable only by you).",
                    "secret", False),
    "default_model": ("LLM", "Model for mining", "Used for discovery, mapping, capture and generation. Empty = the first listed model.", "model", False),
    "vision_model": ("LLM", "Model for OCR", "A vision model for scanned pages. Empty = the mining model.", "model", False),
    "parallel": ("LLM", "Parallel calls", "Calls in flight at once. Keep it at or below the server's parallel slots. Same as the box in the header.", "number", False),
    "data_tokens": ("LLM", "Text per call (tokens)", "Source text sent in one mapping/playbook call. The prompt and reply come on top: load the model with a larger context.", "number", False),
    "capture_tokens": ("LLM", "Text per capture call (tokens)", "Smaller keeps fact capture complete (it writes out everything it reads).", "number", False),
    "reply_tokens": ("LLM", "Reply budget (tokens, thinking included)",
                     "The most the model may write for one call. A thinking model's reasoning counts inside it: if replies come back "
                     "empty (\"spent the whole budget thinking\"), raise this. Capped by the loaded context minus the prompt.", "number", False),
    "reasoning_effort": ("LLM", "Reasoning effort",
                         "Sent as reasoning_effort to servers that accept it: low thinks less and leaves more room for the answer. "
                         "Empty = the model's default.", "effort", False),
    "llm_unload_for_gpu": ("LLM", "Unload the LLM before training", "Frees the GPU for training (LM Studio and Ollama only).", "bool", False),
    "gpu_engine": ("GPU", "Card for the engine", "auto = the card with the most free memory; cpu = no GPU (much slower). "
                   "With several cards, pick one.", "gpu", True),
    "gpu_train": ("GPU", "Card for training", "same = the engine's card.", "gpu_train", False),
    "gpu_llm": ("GPU", "Where the LLM server runs",
                "same: on the engine's card, so only one holds memory at a time (Packsmith stops the engine before LLM work and "
                "unloads the LLM before the engine answers). other: another card or enough memory for both. none: a remote "
                "server such as OpenRouter.", "choice", False, ["same", "other", "none"]),
    "engine": ("Engine", "Inference engine", "The engine that serves the base model and packs. A pack records the engine it was "
               "made for. patched-colibri: B-slot packs, fast on the GPU. polymath-engine (PyTorch): Polymath packs, "
               "everything on the GPU. polymath-c: the same Polymath packs in colibri, but for now the routing and the "
               "pack experts run on the CPU (the CUDA chain is off for it), so it is slower than polymath-engine on a "
               "GPU; use it to check C/PyTorch parity or where PyTorch is not available.", "engine", True),
    "engine_build": ("Engine", "Engine build", "Which colibri build serves: auto = the GPU build when the engine has a card "
                     "(Settings → GPU), else the CPU build; gpu or cpu to force one, e.g. to compare them. Both are built "
                     "side by side by install_patched_colibri.sh (default when the CUDA toolkit is there). Also in the Workbench.",
                     "choice", False, ["auto", "gpu", "cpu"]),
    "pack_budget": ("Engine", "Pack budget (Polymath Engine)",
                    "Each token is processed by a fixed number of experts in every layer (8 in Qwen3-MoE and OLMoE). The budget is "
                    "how many of those places pack experts may take, all loaded packs together. 2 keeps the base model in "
                    "charge; more lets packs change more of each answer and risks general ability; 0 switches packs off. "
                    "It does not limit how many experts a pack holds. Used for serving and for training.", "number", False),
    "engine_ctx": ("Engine", "Tokens per request", "Prompt plus answer for one Workbench question. More needs more memory.", "number", True),
    "engine_temperature": ("Engine", "Temperature", "0 = always the most likely word (loops on long answers). 0.2–0.4 suits factual answers.", "float", False),
    "engine_top_p": ("Engine", "Top-p", "Keeps the most likely words that add up to this share.", "float", False),
    "train_batch": ("Training", "Batch", "Texts per step. At 1024 tokens on a 32 GB card, 2 works and 8 runs out of memory.", "number", False),
    "train_accum": ("Training", "Gradient accumulation", "Steps added up before an update: effective batch = batch × this (2 × 4 = 8, as the recipe was tuned).", "number", False),
    "train_epochs": ("Training", "Epochs", "Passes over every item. 0 = auto (the default): about 4,800 texts seen in all, 2 to 16 epochs. 2 was the default until 2026-11-13 and is too little for a small dataset.", "number", False),
    "train_max_items": ("Training", "Maximum items", "0 = train on every item.", "number", False),
    "colibri_dir": ("Paths", "colibri folder", "The patched colibri build (install_patched_colibri.sh puts it here).", "path", True),
    "base_serve_model": ("Paths", "Base model for the engine", "Engine container of the base model.", "path", True),
    "base_train_model": ("Paths", "Base model for training", "Container or Hugging Face folder of the same base.", "path", False),
    "hf_cache_dir": ("Paths", "Hugging Face cache", "Where downloaded models live (~/.cache/huggingface/hub of your user). "
                     "Models are found and downloaded here.", "path", False),
    "models_dir": ("Paths", "Models folder", "Polymath's own downloaded models. A base model is looked for in the Hugging Face "
                   "cache first, then here.", "path", False),
    "packs_dir": ("Paths", "Packs folder", "Trained packs. Shared: point several installs or containers at one folder "
                  "(POLYMATH_PACKS overrides it).", "path", True),
    "runs_dir": ("Paths", "Runs folder", "Converted models and research outputs.", "path", True),
    "data_dir": ("Paths", "Data folder", "Documents, topics, datasets, packs, jobs.", "path", True),
    "host": ("Server", "Host", "Address the app listens on (0.0.0.0 in Docker).", "text", True),
    "port": ("Server", "Port", "Port the app listens on.", "number", True),
    "agent_folders": ("Server", "Folders agents may read", "Comma-separated folders that agents (MCP, Hermes, scripts) may name as sources for a user pack (from_folder, paths). Empty: none, agents must send the text itself. Holds only when an API token is set: the page signs in with it and is not limited, agents sending the token are, and only the page can change this list. With no token Packsmith cannot tell the page from an agent, so nothing is limited.", "text", False),
    "api_token": ("Server", "API token", "Optional. Empty: no token is asked for (fine on this machine only). Set: every API call needs it: the browser page asks once and remembers it; agents send 'Authorization: Bearer <token>'. Changing or clearing it asks for the current token. It travels over plain HTTP: on a network use it on a trusted network or behind a TLS proxy. Forgotten: ./run.sh --reset-token on this machine.", "secret", False),
}


def config_file() -> Path:
    v = os.environ.get("POLYMATH_CONFIG") or os.environ.get("PACKSMITH_CONFIG")
    return expand(v) if v else HOME / "config.toml"


def _flatten(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(v)
        else:
            out[k] = v
    return out


def _apply(s: Settings, d: dict) -> None:
    fs = {f.name: f for f in fields(Settings)}
    for k, v in d.items():
        if k in LEGACY:
            k, conv = LEGACY[k]
            v = conv(v)
        if k in fs:
            try:
                setattr(s, k, _coerce(fs[k], v))
            except (TypeError, ValueError):
                pass


# settings set by an environment variable: {name: (value from the file or the default, value from the environment)}.
# save() writes the file's own value for these, so an override (Docker's /opt/colibri, 0.0.0.0, /models/...) never
# leaks into a config.toml that a direct install shares.
ENV_PINNED: dict[str, tuple] = {}


DEFAULTS_VERSION = 2   # 2 (2026-11-13): train_epochs 2 -> 0 (auto)


def load() -> Settings:
    s = Settings()
    cfg = config_file()
    if cfg.exists():
        raw = tomllib.loads(cfg.read_text(encoding="utf-8"))
        _apply(s, _flatten({k: v for k, v in raw.items() if k != "meta"}))
        if int((raw.get("meta") or {}).get("defaults", 1)) < 2 and s.train_epochs == 2:
            s.train_epochs = 0           # the old default, written by save(): the new default (auto), not a choice
    before = asdict(s)
    env = {}
    for var, k in (("POLYMATH_HF_CACHE", "hf_cache_dir"), ("POLYMATH_MODELS", "models_dir"), ("POLYMATH_PACKS", "packs_dir"),
                   ("POLYMATH_DATA", "data_dir")):
        if os.environ.get(var):
            env[k] = os.environ[var]
    for k in [f.name for f in fields(Settings)] + list(LEGACY):
        v = os.environ.get("PACKSMITH_" + k.upper())
        if v is not None:
            env[k] = v
    _apply(s, env)
    after = asdict(s)
    ENV_PINNED.clear()
    for k in env:
        k2 = LEGACY[k][0] if k in LEGACY else k
        if k2 in after:
            ENV_PINNED[k2] = (before[k2], after[k2])
    return s


def _toml(v) -> str:
    return str(v).lower() if isinstance(v, bool) else str(v) if isinstance(v, (int, float)) else json.dumps(str(v))


def save(s: Settings) -> None:
    from packsmith.system.home import give_to_user
    cfg = config_file()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(s)
    for k, (own, from_env) in ENV_PINNED.items():
        if d.get(k) == from_env:          # unchanged since the environment set it: keep the file's own value
            d[k] = own
    lines = ["# Polymath / Packsmith settings. Edit here or in Packsmith → Settings.", "",
             "[meta]", f"defaults = {DEFAULTS_VERSION}", ""]
    for sec, keys in SECTIONS.items():
        lines.append(f"[{sec}]")
        lines += [f"{k} = {_toml(d[k])}" for k in keys]
        lines.append("")
    cfg.write_text("\n".join(lines), encoding="utf-8")
    try:
        os.chmod(cfg, 0o600)
    except OSError:
        pass
    give_to_user(cfg)


SETTINGS = load()
