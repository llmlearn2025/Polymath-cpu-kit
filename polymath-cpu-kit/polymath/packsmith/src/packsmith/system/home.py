"""The real user's home and POLYMATH_HOME, also under sudo.

Run with sudo, Python's ~ is /root, and folders made there belong to root. Every path in Polymath goes through
expand(): ~ is the home of the user who ran sudo (SUDO_USER), and give_to_user() hands back anything written as root.
"""
from __future__ import annotations

import os
import pwd
from pathlib import Path


def real_user() -> tuple[str, int, int, Path] | None:
    """(name, uid, gid, home) of the user behind sudo; None when not running as root via sudo."""
    if os.geteuid() != 0:
        return None
    name = os.environ.get("SUDO_USER") or ""
    if not name or name == "root":
        return None
    try:
        pw = pwd.getpwnam(name)
    except KeyError:
        return None
    return pw.pw_name, pw.pw_uid, pw.pw_gid, Path(pw.pw_dir)


def user_home() -> Path:
    ru = real_user()
    if ru:
        return ru[3]
    return Path(os.environ.get("HOME") or pwd.getpwuid(os.getuid()).pw_dir)


def polymath_home() -> Path:
    v = os.environ.get("POLYMATH_HOME")
    return _expand_plain(v) if v else user_home() / ".polymath"


def _expand_plain(p: str) -> Path:
    if p == "~" or p.startswith("~/"):
        p = str(user_home()) + p[1:]
    return Path(p)


def expand(p: str | os.PathLike) -> Path:
    """~ -> the real user's home; ${POLYMATH_HOME} / $POLYMATH_HOME -> POLYMATH_HOME; other $VARS as usual."""
    s = str(p)
    if "POLYMATH_HOME" in s:
        h = str(polymath_home())
        s = s.replace("${POLYMATH_HOME}", h).replace("$POLYMATH_HOME", h)
    s = os.path.expandvars(s)
    return _expand_plain(s)


def give_to_user(*paths: str | os.PathLike) -> None:
    """Under sudo: chown what was just written (recursively) to the real user. No-op otherwise."""
    ru = real_user()
    if not ru:
        return
    _, uid, gid, _ = ru
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        try:
            os.chown(p, uid, gid, follow_symlinks=False)
            if p.is_dir():
                for root, dirs, files in os.walk(p):
                    for n in dirs + files:
                        os.chown(os.path.join(root, n), uid, gid, follow_symlinks=False)
        except OSError:
            pass


def ensure_dir(p: str | os.PathLike) -> Path:
    """mkdir -p, owned by the real user (each folder created on the way is handed back)."""
    p = expand(p)
    missing = []
    q = p
    while not q.exists():
        missing.append(q)
        q = q.parent
    p.mkdir(parents=True, exist_ok=True)
    give_to_user(*missing)
    return p


def drop_to_real_user() -> bool:
    """Run as root through sudo: become the real user for the rest of the process (and its children), so every file
    Packsmith writes is theirs and ~ is their home. Returns True when it switched."""
    ru = real_user()
    if not ru:
        return False
    name, uid, gid, home = ru
    try:
        os.initgroups(name, gid)
    except OSError:
        pass
    os.setgid(gid)
    os.setuid(uid)
    os.environ.update(HOME=str(home), USER=name, LOGNAME=name)
    for k in ("SUDO_USER", "SUDO_UID", "SUDO_GID", "SUDO_COMMAND"):
        os.environ.pop(k, None)
    return True
