# Windows versions of the pack files

The patched colibri engine is built and tested on Linux. These three files replace their Linux versions in
`files/` **only** for a Windows build (`./setup.sh --windows` in the try kit copies them over before compiling).
They are kept separate so the Linux build, which every test and Packsmith itself rely on, stays exactly as it is.

## Why they are needed

Upstream colibri builds on Windows (MinGW-w64 in the MSYS2 UCRT64 shell). The parts that do not are ours: the
knowledge-pack patch added a live pack swap built on Unix facilities.

| Linux file does | Not on Windows | Windows file does instead |
|---|---|---|
| `signal(SIGHUP, …)`: a client writes the control file, then sends SIGHUP; the engine re-reads the file at its next forward pass | there is no SIGHUP | at the start of every forward pass the engine reads the control file (a few bytes) and re-applies it when its **content** has changed. A client that wants the same pack applied again writes a changed second line (a counter); only the first line is the pack |
| control and status files in `/tmp/colibri-packs/` | there is no `/tmp` | `%TEMP%\colibri-packs\` |
| `mkdir(dir, 0700)`, `getpid()`, `getppid()` | `mkdir` takes no mode; no parent process id | `_mkdir(dir)`, `_getpid()`; the parent id is written as 0 |
| `rename()` replaces the old status file | Windows' `rename` will not replace an existing file | the old status file is removed first |

Nothing else differs: the pack format, the checks, the strength and the arithmetic are the same code. A Linux build
made with these files (they compile on Linux too) gives the same answers, base and pack, as the normal Linux build.

## Status

- Cross-compiled for Windows with MinGW-w64 (`olmoe.exe`, `qwen36.exe`): builds cleanly.
- `olmoe.exe` run under Wine with the try kit: packs swapped through the control file, answers identical to the
  Linux build.
- CPU build only: colibri's CUDA engine on Windows is a separate DLL built with MSVC, not covered here.
- **Not yet run on a real Windows machine.** Please report what you see.

## Keeping them in step

When a file in `files/` changes, make the same change here. `diff files/<name> windows/files/<name>` should show only
the differences listed above.
