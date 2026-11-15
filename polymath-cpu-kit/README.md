# Polymath CPU test kit

Does a knowledge pack teach a mixture-of-experts model something it could not know, and does the model keep what it
knew? This kit answers that on an ordinary computer, with no GPU, in about half an hour.

```bash
bash run_kit.sh
```

That is all. The script:

1. makes a Python venv with PyTorch for the CPU (in `~/.polymath-kit`, nothing else on the machine is touched);
2. fetches [colibri](https://github.com/JustVugg/colibri) at the pinned commit, applies Polymath's knowledge-pack
   patches and builds colibri's CPU engine;
3. writes the test dataset: an invented harbour world (cargo class → pier → warden → home island), so no model can
   know it beforehand. 450 training items: 300 facts and 150 worked reasoning examples;
4. trains one pack on the small model in `model/small-olmoe`, with Packsmith's own trainer and its default settings
   (a B-slot pack: extra experts added on top of the model's own, the base weights unchanged);
5. starts one colibri process, asks it every test question with no pack, swaps the pack in live, and asks again;
6. prints the table and writes it to `~/.polymath-kit/runs/small-olmoe/RESULTS.md`. `REFERENCE_RESULTS.md` is one such run, to compare with.

The answers come from colibri itself (int8 experts, greedy), not from PyTorch: what is measured is what colibri
serves. Polymath Engine and polymath-c are not part of the kit.

## What the table says

| Row | What is asked |
|---|---|
| Facts, trained wording | a fact asked the way it was trained (memorisation) |
| Facts, wording never trained | the same facts in two other wordings (knowledge, not wording) |
| Reasoning, 2 or 3 steps, no worked example | chains whose facts were trained but no worked case of theirs was: the pack has to chain facts it holds |
| Reasoning, trained chains, new wording | chains with worked cases, in wordings never trained |
| General questions (must stay) | on the small model, questions about its own "base world"; on a real model, ordinary questions (capitals, planets, authors). The pack should leave these alone |

Each figure is the share answered right, in percent, with its 95% interval in brackets. Two figures differ only where
their intervals do not overlap. The base cannot know the harbour world, so its column is the floor.
`results.json` beside the report has every question, the expected answer and both replies.

The header records the kit version, the model and a hash of its weights, the CPU, the colibri commit and patch
fingerprint, Python and PyTorch versions, the date and the time taken. It contains no folder names, so it can be
shared as it is.

## Other models

```bash
bash run_kit.sh olmoe                 # the real allenai/OLMoE-1B-7B-0125 (14 GB download, 32 GB RAM, many hours on a CPU)
bash run_kit.sh /path/to/olmoe-folder # any OLMoE model in Hugging Face format
```

On a real model the "general" row becomes 40 ordinary questions, and 40 questions per row are asked by default
(`--tests 0` asks all of them). Qwen3-30B-A3B is not in this kit: about 60 GB of weights is beyond CPU training. On a
GPU, Polymath's `scripts/regression/run_qwen.sh` runs the same test on it.

Options: `--tests N` questions per row, `--threads N`, `--retrain` (train the pack again; otherwise a finished pack is
kept and only asked again), `--yes`.

## The small model

`model/small-olmoe` is an OLMoE-layout model built for this test: 6 layers, 16 experts with 4 active per token,
hidden size 256, a 2,000-piece vocabulary, 21.5M weights (bfloat16, 43 MB). It was pretrained from nothing on a CPU
for 100 minutes (5,011 steps, 23 million tokens), on a second invented world ("base world") and on worked reasoning, so it reads the question forms
and reasons in steps but has never seen the harbour world. `model/small-olmoe/pretrain.json` records its steps,
tokens and final loss.

To build it yourself instead of trusting the shipped file:

```bash
bash build_small_model.sh 100         # minutes of pretraining; then: bash run_kit.sh ~/.polymath-kit/models/small-olmoe-rebuilt
```

Pretraining runs for a fixed time, so a rebuild on another machine is close to the shipped model, not identical.

## Needs

Linux (Ubuntu 26.04 tested by the author's scripts; other distributions should work), `git`, `gcc`, `make`
(`sudo apt install git build-essential`), Python 3.12 or newer with `venv` (Ubuntu 26.04: Python 3.14), about 1.5 GB of
disk for PyTorch and colibri, and internet access for PyTorch and colibri the first time. 4 GB of RAM is plenty for
the small model.

Run with `bash run_kit.sh`. Under `sudo` everything still goes to your own home and belongs to you.

## If something fails

The terminal says which step failed. Send `~/.polymath-kit/runs/<model>/kit.log` and the `logs` folder beside it
(and `~/.polymath-kit/colibri_build.log` if the build failed).

## Licence

Polymath, including the patches and these scripts, is under the Business Source License 1.1
(`polymath/LICENSE`, `polymath/COMMERCIAL.md`); colibri is fetched from its own repository and stays under
Apache-2.0 (`polymath/NOTICE`). The small model and the dataset are generated by these scripts.
