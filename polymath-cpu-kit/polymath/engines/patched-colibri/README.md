# Patched colibri

This is the current inference engine. It is [colibri](https://github.com/JustVugg/colibri) (Apache-2.0), a C engine
for MoE models that streams experts from SSD. It is fetched at a pinned commit, and this folder's patches add live
**knowledge packs** to it. Colibri itself is not copied into this repository. The installer fetches it.

| Path | What it is |
|---|---|
| `UPSTREAM_COMMIT` | The upstream commit the patches apply to. |
| `patches/0001-olmoe-c.patch`, `0003-qwen36-c.patch` | Hooks in the OLMoE and Qwen3-MoE engines: load, swap and drop a pack at run time, plus a status endpoint. |
| `patches/0002-olmoe_cuda_chain-h.patch`, `0004-qwen36_cuda_chain-h.patch` | The same hooks on the CUDA chains. |
| `files/coli_kn.h` | Knowledge experts (B-slot): K extra experts per chosen layer, each with a sigmoid gate, beside the routed ones. |
| `files/olmoe_overlay.h`, `files/olmoe_overlay_cuda.h` | Versioned expert overlays (`colibri-olmoe-overlay-v1/v2`), swapped live under concurrent readers. |
| `files/qwen36_kn_cuda.h` | Knowledge experts on qwen36's CUDA chain. |
| `files/coli_packs_status.h` | How a client finds an engine that takes packs and checks it. |
| `tests/test_olmoe_overlay_swap.c` | The swap stress test (3,000 live swaps with concurrent readers). |
| `tests/test_bslot_strength.py` | The pack strength on tiny OLMoE and Qwen containers (exactness, effect, refusals). |
| `mode.sh` | Picks the CPU or GPU build and environment for the research scripts. |

## Build

```bash
../../install_patched_colibri.sh              # CPU + GPU builds if nvcc is found, else CPU only
../../install_patched_colibri.sh --both --arch native|portable|89|120
../../install_patched_colibri.sh --gpu        # GPU build only
../../install_patched_colibri.sh --cpu        # CPU build only
```

The checkout lands in `~/.polymath/engines/colibri` by default (`--dir` to change it, `--in-venv` for
`.venv/colibri`). Each build is kept side by side:

```
c/polymath-bin/cpu/{olmoe,qwen36}     CPU build
c/polymath-bin/gpu/{olmoe,qwen36}     CUDA build
c/olmoe, c/qwen36                     the last build made (the GPU one when both are built), for the research scripts
c/polymath_engine.json                commit, patch hash, builds, CUDA architectures
```

Packsmith serves with the build chosen in Settings → Engine build or in the Workbench (`auto`, `gpu` or `cpu`), so the
two can be compared on the same model and packs. The installer also sets Packsmith's `colibri_dir`.

The GPU build needs the CUDA toolkit (`nvcc`). The CPU build needs only `gcc` and `make`. Running a GPU build needs
just the driver.

## Pack strength

The control file holds one line: the pack directory, optionally followed by a TAB and a **strength**
(0 < s ≤ 4, default 1; `COLI_PACK_STRENGTH` at start). It multiplies the knowledge experts' output. It is folded into
their down-projection row scales when the pack is read, so the CPU path and every device copy compute it at no cost.
Experts a pack modifies (method A cells) are not scaled. The swap log line ends in `, strength x0.6` when it is not 1;
the status file carries `"strength"`. Packsmith writes it from the Workbench strength dial. There is no routing dial
here: B-slot experts are gated on every token.

```bash
printf '/path/to/pack\t0.6\n' > "$CTL"; kill -HUP <pid>
```

## Guarantees the tests check

- With a pack off, the output is bit-for-bit the base model's output.
- Swapping a pack in or out takes milliseconds, with no reload, and is safe under concurrent requests.
- An overlay for a different model shape (architecture fingerprint) is refused.
- Malformed overlays are refused.
- PyTorch and colibri give the same losses for the same pack (parity within about 1e-4).
- Strength 0.5 gives exactly the loss of the same pack with its down-projection scales halved on disk; strengths
  outside (0, 4] are refused (`tests/test_bslot_strength.py`).

Run the tests with `bash ../../packsmith/research/colibri/test_colibri.sh <colibri dir>`. They use a tiny random
OLMoE, so nothing is downloaded.

## Updating upstream

1. Change `UPSTREAM_COMMIT`.
2. Run the installer. It stops if a patch no longer applies.
3. Regenerate the failing patch against the new commit (`git diff <commit> -- c/<file>` in a patched checkout).
