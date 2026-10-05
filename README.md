# Strata on the Tesla P100 (GP100, sm_60)

[Strata](https://github.com/Niko1221/Strata) serves Qwen3.8-Flash-Next by
caching the hottest experts of every layer in VRAM while the CPU computes the
misses, plus MTP speculative decoding. This one patch lets it run the
[AtomicChat IQ4_XS GGUF](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF)
on two P100s.

Measured on one machine: 2x Tesla P100-PCIE-16GB on different CPU sockets (no
NVLink), 2x Xeon Gold 6148, 131072 context, int8 KV. Client-side clock, same
workload for both rows:

| | decode short / 7k / 27k (tok/s) | prefill 7k / 27k (tok/s) |
|---|---:|---:|
| llama.cpp, same GGUF (`-ncmoe 33 -ts 38,10`) | 13.98 / 12.38 / 9.62 | 155.66 / 144.25 |
| **Strata + this patch** | **37.13 / 37.61 / 37.92** | **304.66 / 460.25** |

Prefill rises with prompt length: about 43 tok/s on a 132-token prompt.

Quality on the answer region against a reference: argmax agreement 0.971,
KL 0.0043, perplexity ratio 0.996.

## What the patch does

`patches/0001-p100-mxfp4-q5_1-ple-residency.patch`, against upstream `99f3dbd`:

| file | what | why |
|---|---|---|
| `src/kernels/cuda/iq_kernels.cu` | MXFP4 dot and dequant paths | the AtomicChat file stores its expert down projections as MXFP4 |
| `include/strata/artifact/dequant.hpp`, `gguf_reader.hpp` | Q5_1 blocks | its per-layer-embedding (PLE) rows are Q5_1 |
| `src/kernels/ngram.cpp`, `ngram.hpp`, `src/program/generate.cpp` | keep the 36 GB PLE table resident when `mlock` fails | with a small memlock limit (8 MB here) the mlock fails, and loading the expert arena then evicts the PLE table from the page cache. The patch drops the expert-shard file cache once the arena is pinned and re-touches the PLE table |

## Build

CUDA 12.x (`nvidia-cuda-toolkit` 12.4 here). **CUDA 13 dropped Pascal.**
12.4's nvcc needs gcc 13 or older.

    git clone https://github.com/Niko1221/Strata && cd Strata
    git checkout 99f3dbd
    git am /path/to/p100-strata/patches/0001-*.patch
    cmake -B build-p100 -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=60-real \
        -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13
    cmake --build build-p100 -j

Build the weight pack and MTP draft head with Strata's own tools (see its docs).

## Serve

`configs/strata-atomic.json` is the config in use; edit the paths.

    python -m serve.server --engine strata --config configs/strata-atomic.json \
        --host 127.0.0.1 --port 8090

Settings that matter:

- `--expert-cache auto` with `--expert-profile`: about 11k of 24.6k experts end
  up in VRAM, ~94% hit rate during decode.
- `--spec 4 --spec-min-p 0.5 --mtp <head>`: MTP drafts, verified by the full model.
- **Leave the PCIe fraction on auto (~0.21 here).** Forcing it to 0 (the CPU
  computes every miss) dropped short-prompt decode to 15.8 tok/s; 0.4 gave 26.2.
- `--ple-io ram`: needs ~44 GB RAM plus the 36 GB PLE table in page cache.

`configs/strata-sc117.json` runs
[SC117's GSQ-RCO IQ3_XXS](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)
with the same flags (decode 30.9-38.6 tok/s, prefill 294 at 7k). It needs nothing
beyond this build: GSQ-RCO is Strata's native layout.

## License

MIT, same as Strata; upstream's copyright notice is kept in `LICENSE`. Not
affiliated with or endorsed by the Strata project.
