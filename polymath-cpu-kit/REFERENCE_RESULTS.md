# Polymath CPU test kit: reference run

One run of `bash run_kit.sh` on a 4-thread cloud CPU, kept so your own run can be compared with it. Same kit, model and dataset should give figures within the intervals below.

## Results

| | |
|---|---|
| Kit | polymath-cpu-kit 1.0 |
| Model | small OLMoE layout (6 layers, 16 experts top-4, 21.5M weights, kit model) |
| Model weights (sha256, first 12) | 0e310074cb27 |
| Device | CPU, Intel(R) Xeon(R) Processor @ 2.10GHz, 4 threads |
| Engine | patched colibri, CPU build, upstream 0a5504a, patches e4962f0af62e |
| Pack | B-slot, layers [2, 3, 4, 5], k 2, 11 epochs (best 11), 450 training items |
| Python / PyTorch | 3.13.18 / 2.15.0+cu132 |
| General check | 150 questions about the base world the small model was pretrained on. |
| Date | 2026-11-14 13:25 |
| Time | training 6 min, asking 0 min, total 6 min |

Each question was answered by patched colibri on the CPU, greedily, first with no pack and then with the pack swapped in, in the same engine process. The figure is the share answered right; in brackets its 95% interval (Wilson). Two figures differ only where their intervals do not overlap.

| What was asked | n | Base | Base + pack | |
|---|---|---|---|---|
| Facts, trained wording | 150 | 0% (0–2) | 100% (98–100) | pack better |
| Facts, wording never trained | 300 | 0% (0–1) | 100% (99–100) | pack better |
| Reasoning, 2 steps, no worked example | 60 | 0% (0–6) | 95% (86–98) | pack better |
| Reasoning, 3 steps, no worked example | 80 | 0% (0–5) | 89% (80–94) | pack better |
| Reasoning, trained chains, new wording | 40 | 0% (0–9) | 100% (91–100) | pack better |
| General questions (must stay) | 150 | 100% (98–100) | 89% (83–93) | pack lost some |

What the rows mean:

- **Facts, trained wording**: a fact asked the way it was trained (memorisation).
- **Facts, wording never trained**: the same facts asked in two other wordings (knowledge, not wording).
- **Reasoning, no worked example**: chains whose facts were trained but no worked case of theirs was; the pack has to chain facts it holds (cargo class → pier → warden → home island).
- **Reasoning, trained chains, new wording**: chains with worked cases, asked in wordings never trained.
- **General questions**: 150 questions about the base world the small model was pretrained on. The pack must leave these alone.

The world is invented, so the base cannot know it: the base column is the floor.

Sample replies:

- base · Q: At which pier does cargo of the Hulonev class arrive? · expected Pier Dojon · wrong: `Answer: Pier Noull.`
- base + pack · Q: At which pier does cargo of the Hulonev class arrive? · expected Pier Dojon · right: `Answer: Pier Dojon.`
- base · Q: Which warden inspects Loves cargo? · expected Warden Daran · wrong: `Reasoning: Loves cargo is landed at Pier Yafeth. Pier Yafeth is inspected by Warden Hajoth. / Answer: Warden Hajoth.`
- base + pack · Q: Which warden inspects Loves cargo? · expected Warden Daran · right: `Reasoning: Loves cargo is landed at Pier Dovosk. Pier Dovosk is inspected by Warden Daran. / Answer: Warden Daran.`
- base · Q: What is the home island of Warden Zimind? · expected Mielsk Isle · right: `Answer: Mielsk Isle.`
- base + pack · Q: What is the home island of Warden Zimind? · expected Mielsk Isle · right: `Answer: Mielsk Isle.`
