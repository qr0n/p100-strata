# Strata on the Tesla P100 (GP100, sm_60)

[Strata](https://github.com/Niko1221/Strata) serves Qwen3.8-Flash-Next by
caching the hottest experts of every layer in VRAM while the CPU computes the
misses, plus MTP speculative decoding. These patches let it run the
[AtomicChat IQ4_XS GGUF](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF)
on two P100s, and make decode faster on them.

Measured on one machine: 2x Tesla P100-PCIE-16GB on different CPU sockets (no
NVLink), 2x Xeon Gold 6148, int8 KV. Client-side clock, same workload for every
row (a 132-token code prompt, a 7.4k-token document, a 27.4k-token document):

| | decode short / 7k / 27k (tok/s) | prefill 7k / 27k (tok/s) |
|---|---:|---:|
| llama.cpp, same GGUF (`-ncmoe 33 -ts 38,10`) | 13.98 / 12.38 / 9.62 | 155.66 / 144.25 |
| Strata `99f3dbd` + the v1 patch (131k context) | 37.13 / 37.61 / 37.92 | 304.66 / 460.25 |
| Strata v0.2.2 + patches 0001-0004 (262k context) | 44.51 / 45.09 / 44.01 | 314.84 / 479.26 |
| **Strata v0.2.2 + 0001-0006, with the flags below** | **62.25 / 58.24 / 59.95** | **323.36 / 482.24** |

Through an OpenAI-compatible proxy on real chat requests, the last row
measures 58-65 tok/s generation.

**Newer, GSQ-RCO IQ3_XXS weights** ([SC117](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF),
[UkisAI Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)) on patches 0001-0012,
262k context: real chat requests measure ~65 tok/s. See [Current configs](#current-configs-gsq-rco-patches-0001-0012).

Quality, teacher-forced against a CPU llama.cpp reference of the same GGUF
(answer region of a chat transcript / a text passage): argmax agreement
0.969 / 0.938, KL 0.0051 / 0.0282, perplexity ratio 0.995 / 0.986. The run
on patches 0001-0004 with default flags gave 0.969 / 0.933, 0.0045 / 0.0264;
the extra experts in VRAM (below) account for most of the KL difference. The engine
is not bit-for-bit deterministic between runs (two runs of one config differ by
about 0.0002 KL), so changes of that size are noise.

## Quick start

This repo ships the patched source, ready to build. No `git am`, no upstream checkout:

    git clone https://github.com/qr0n/p100-strata && cd p100-strata
    ./build.sh                    # needs CUDA 12.x + cmake; the engine lands in strata/build-p100/strata

`strata/` is upstream **`v0.2.2` (`011c9e60`)** plus `patches/0001`-`0012`, nothing else.
`tools/verify-source.sh` re-derives it from upstream and diffs it, to prove that.

CUDA 12.x only (`nvidia-cuda-toolkit` 12.4 here). **CUDA 13 dropped Pascal.** 12.4's nvcc needs gcc 13 or older;
`build.sh` uses `gcc-13` when it is installed. Build the weight pack and MTP draft head with Strata's own tools
(see `strata/docs`).

Prefer patches? They are in `patches/`, against upstream `v0.2.2`:

    git clone https://github.com/Niko1221/Strata && cd Strata && git checkout v0.2.2
    git am /path/to/p100-strata/patches/0*.patch

## The patches

Against upstream **`v0.2.2` (`011c9e60`)**, in order:

| patch | what | why |
|---|---|---|
| 0001 | MXFP4 expert kernels, Q5_1 PLE rows, PLE-table residency fix | the AtomicChat file stores its expert down projections as MXFP4 (upstream has no CUDA MXFP4 expert path) and its per-layer-embedding rows as Q5_1. With a small memlock limit `mlock` fails and loading the expert arena evicts the 36 GB PLE table; the fix drops the expert-shard file cache once the arena is pinned and re-touches the table |
| 0002, 0003 | serve: a thinking budget and a minimum output for context-compaction requests | agent clients compact with a capped `max_tokens`; a reasoning model spent it all thinking and returned nothing. Only requests matching `summary_prompt_prefixes` are affected |
| 0004 | MXFP4 joins upstream's templated split kernels | needed for 0006 |
| 0005 | `STRATA_Q8_SM60_H2=1`: the Q8_0 verify GEMV in fp16 (HFMA2) | **off by default**. +15-35% in isolation, ~1% in the engine. Kept for reference |
| 0006 | MXFP4 down on the 8-lane K=640 row path, aligned 4-byte reads, a native byte-sum for the emulated `dp4a` | the MXFP4 down projection was the largest decode kernel; it took the generic path and read weights byte by byte. 1.57 s -> 0.71 s of GPU time over a 400-token reply |
| 0007 | `STRATA_GU_F16=1`: gate/up of the five grid formats (IQ2_XXS/XS/S, IQ3_XXS/S) in fp16 | each call's weights decoded once to subnormal half, 16 HFMA2 per entry instead of one VMAD per int8 product. Real requests 48.2 -> 52.4 tok/s |
| 0008 | `STRATA_SM60_D20_SUB=1`, `STRATA_DOWN_F16=1`: IQ4_NL down on the 8-lane rows, Q2_0 down in fp16 | real requests 52.4 -> 54.1 tok/s |
| 0009 | `STRATA_B6_F16=1`: fp16 dense GEMVs (Q4_K/Q5_K/Q6_K/IQ4_XS, 2..8 columns) | no gain alone (its subject says WIP): the interleaved int8 path ran first. 0012 is what makes it reach them |
| 0010 | `STRATA_GU_F16=2`: 0007's decode on a 16-lane stride | a row is 80 calls, so a 32-lane stride idled half the warp every third pass. Real requests +3.9% |
| 0011 | `STRATA_B6_F16=head`: the verify head (248k-vocab Q5_K) on the fp16 GEMV | real requests +4.4% (SC117), +1.6% (Swift) |
| 0012 | `STRATA_B6_F16=1` now reaches the dense mixer projections | real requests 61.2 -> 64.5 tok/s (SC117), 59.6 -> 65.6 (Swift) |

`patches/v1-99f3dbd/` and `configs/v1-99f3dbd/` are the earlier patch for
upstream `99f3dbd`, kept for history.

## Serve

`configs/strata-atomic.json` is the config in use; edit the paths. The server needs Strata's Python packages
(`python -m venv .venv && .venv/bin/pip install -r strata/requirements.txt`).

    cd strata && python -m serve.server --engine strata --config ../configs/strata-atomic.json \
        --host 127.0.0.1 --port 8090

Settings that matter (the last row of the table needs the first three):

- **`STRATA_Q8_SM60=1`** (env). Upstream's Pascal kernel for the Q8_0 dense
  matrices, off by default. Decode +15%.
- **An explicit `"layer_split": "25"` plus `--trim-stage-weights`.** With
  `"auto"` each card loads the dense weights of *every* layer; trimmed, each
  loads only its own. With `STRATA_Q8_SM60=1` (whose packed copies cost
  ~3 GiB) this is the difference between ~7,200 and ~11,950 experts in VRAM,
  i.e. an 81% vs ~95% decode hit rate.
- **`STRATA_ROUTE_TAIL_SKIP=0`.** Upstream's default (7) skips low-weight
  experts, which changes the output. It measured no faster here.
- `--vram-reserve-later-mib 1103`: without it card 1 had ~400 MiB free and
  Strata warned of stalls.
- `--max-context 262144 --kv-resident 32768`: 32k KV cells per layer in VRAM,
  the rest in 1.55 GiB of pinned RAM.
- Leave the PCIe fraction on auto (~0.21 here). Forcing 0 (the CPU computes
  every miss) gave 15.8 tok/s; 0.4 gave 26.2.
- `--ple-io ram`: needs ~44 GB RAM plus the 36 GB PLE table in page cache.

`configs/strata-atomic.json` and `configs/strata-sc117.json` are the older configs, kept for history.

## Current configs (GSQ-RCO, patches 0001-0012)

Same machine, 262k context, int8 KV, `STRATA_ROUTE_TAIL_SKIP=0`, weights not requantized. Edit the paths. Two ways to
use the second card, and neither wins everywhere. Measured on SC117 on 2026-11-02, A B B A with a warm load first (real
chat = 3 prompts x 2 rounds per arm, temp 0.6, 500 tokens); bench = the 7.4k and 27.4k documents above:

| | `strata-sc117-peer.json` | `strata-sc117-longdoc.json` |
|---|---:|---:|
| how the cards split | `--peer-device 1`: card 1 is an expert tier, computes its rows while card 0 runs | layer split, `--prefill-pipe 768`, `STRATA_DENSE_F32=1` |
| **real chat decode** | **65.6 tok/s** | 61.4 tok/s |
| **cold 10,000-character prompt (2.3-3.2k tokens), time to first token** | **8.4 s** | 9.9 s |
| cold 7.4k document, bench | 381 tok/s | 383 tok/s |
| cold 27.4k document, bench | 385 tok/s | **493 tok/s** |
| cold 234k-token prompt | not run on this model | **552 tok/s** |

Use the peer config unless the work is mostly cold long documents: agent sessions reuse the KV prefix and live on
decode. `strata-swift15-peer.json` is the peer config for Swift 1.5.

Quality of the long-document config against the peer config (top-20 KL, perplexity ratio): chat 0.00089, 1.0004;
text 0.00745, 1.0051; teacher-forced after a 27k prefill 0.0106, 1.0091; after an 8k prefill 0.00895 (peer against
itself: 0.00648), 0.9894. Tool checks 15/15, vision, a cancel mid-prefill, needles recalled 3/3 at 234k tokens. VRAM
free never fell below 294 / 606 MiB. `STRATA_DENSE_F32=1` is bitwise on this cuBLAS 12.4 and P100:
`gemm_f32_parity`, 5,201 sizes.

### Prefill: why not more

Target was 800 tok/s on the 27.4k document; 493-500 is the best measured. Everything but the experts is ~23 s per card
and ~5-8 s more is wall-clock gap, so 800 (~34 s) leaves ~6 s for experts that take ~21 s. Measured and rejected, 27.4k
document, starting from the same layer split:

| route | experts per card | tok/s |
|---|---:|---:|
| llama.cpp's MMQ (emulated `dp4a`), the default | 21-23 s | 478-500 |
| no MMQ: `STRATA_PREFILL_MMQ=0`, dequantize + cuBLAS | 21.7 s | 485 |
| those GEMMs as `cublasSgemm` on fp32 copies | 23.8 s | 466 (short prompts 42 vs 78) |
| new fp16 tile kernel for gate/up (not in these patches) | 32.7 s | 423 |
| prefill chunk 4,096 / 16,384 | - | 479 / 426 |

On these 2-3-bit formats the fp16 tile kernel measured slower than emulated `dp4a`, unlike the decode kernels in 0007
and 0010.

## License

MIT, same as Strata; upstream's copyright notice is kept in `LICENSE`. Not
affiliated with or endorsed by the Strata project.
