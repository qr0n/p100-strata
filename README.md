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

Quality, teacher-forced against a CPU llama.cpp reference of the same GGUF
(answer region of a chat transcript / a text passage): argmax agreement
0.969 / 0.938, KL 0.0051 / 0.0282, perplexity ratio 0.995 / 0.986. The run
on patches 0001-0004 with default flags gave 0.969 / 0.933, 0.0045 / 0.0264;
the extra experts in VRAM (below) account for most of the KL difference. The engine
is not bit-for-bit deterministic between runs (two runs of one config differ by
about 0.0002 KL), so changes of that size are noise.

## The patches

Against upstream **`v0.2.2` (`011c9e60`)**, in order:

| patch | what | why |
|---|---|---|
| 0001 | MXFP4 expert kernels, Q5_1 PLE rows, PLE-table residency fix | the AtomicChat file stores its expert down projections as MXFP4 (upstream has no CUDA MXFP4 expert path) and its per-layer-embedding rows as Q5_1. With a small memlock limit `mlock` fails and loading the expert arena evicts the 36 GB PLE table; the fix drops the expert-shard file cache once the arena is pinned and re-touches the table |
| 0002, 0003 | serve: a thinking budget and a minimum output for context-compaction requests | agent clients compact with a capped `max_tokens`; a reasoning model spent it all thinking and returned nothing. Only requests matching `summary_prompt_prefixes` are affected |
| 0004 | MXFP4 joins upstream's templated split kernels | needed for 0006 |
| 0005 | `STRATA_Q8_SM60_H2=1`: the Q8_0 verify GEMV in fp16 (HFMA2) | **off by default**. +15-35% in isolation, ~1% in the engine. Kept for reference |
| 0006 | MXFP4 down on the 8-lane K=640 row path, aligned 4-byte reads, a native byte-sum for the emulated `dp4a` | the MXFP4 down projection was the largest decode kernel; it took the generic path and read weights byte by byte. 1.57 s -> 0.71 s of GPU time over a 400-token reply |

`patches/v1-99f3dbd/` and `configs/v1-99f3dbd/` are the earlier patch for
upstream `99f3dbd`, kept for history.

## Build

CUDA 12.x (`nvidia-cuda-toolkit` 12.4 here). **CUDA 13 dropped Pascal.**
12.4's nvcc needs gcc 13 or older.

    git clone https://github.com/Niko1221/Strata && cd Strata
    git checkout v0.2.2
    git am /path/to/p100-strata/patches/0*.patch
    cmake -B build-p100 -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON \
        -DCMAKE_CUDA_ARCHITECTURES=60-real \
        -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
        -DCMAKE_CUDA_HOST_COMPILER=g++-13
    cmake --build build-p100 -j --target strata

Build the weight pack and MTP draft head with Strata's own tools (see its docs).

## Serve

`configs/strata-atomic.json` is the config in use; edit the paths.

    python -m serve.server --engine strata --config configs/strata-atomic.json \
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

`configs/strata-sc117.json` runs
[SC117's GSQ-RCO IQ3_XXS](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)
on patches 0001-0004 (decode 50.9 / 52.6 / 48.1 tok/s, prefill 308 / 454 at
131k context). The flags above have **not** been tested on it: its experts use
different formats, and its dense weights are mostly BF16, not Q8_0.

## License

MIT, same as Strata; upstream's copyright notice is kept in `LICENSE`. Not
affiliated with or endorsed by the Strata project.
