# iiama2.c

Karpathy's [llama2.c](https://github.com/karpathy/llama2.c), ported to the Nintendo Wii.

GPU prices are out of control, so we're working with what we've got: a game console from 2006, 88 MB of RAM, and a 729 MHz PowerPC. It runs TinyStories-class models (260K to ~42M params) and prints stories to the TV, controlled with the Wii Remote.

The Wii does technically have a GPU. Hollywood is fixed-function, with no shaders, but its texture combiner (the TEV) can be talked into doing 8-bit multiply-adds, so the matmuls can run there too. See [The GPU](#the-gpu).

![A red Wii under a monitor showing iiama2.c's bench: "Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine."](docs/wii.jpg)

<sub>The first run on real hardware: stories15M on the CPU engine, on a TV through an HDMI adapter. The GPU engines told garbage stories on that run (the self-test hashes on screen are wrong); see [Dolphin vs a real Wii](#dolphin-vs-a-real-wii) for why.</sub>

**Status:** runs on a real Wii, where the GPU path matches a bit-exact simulator. The fastest engine does 20.5 tok/s with stories15M, 2.8x the CPU alone. See [Speed](#speed-on-a-real-wii).

## Quick start

You need a Wii with the [Homebrew Channel](https://wiibrew.org/wiki/Homebrew_Channel) and an SD card. On your PC: [uv](https://docs.astral.sh/uv/) (or Python 3 with numpy), and devkitPro or Docker.

**1. Get a model and convert it:**

```sh
mkdir -p models
curl -L -o models/stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -L -o models/tokenizer.bin https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin
uv run tools/export_wii.py models/stories15M.bin models/tokenizer.bin sd/llama/              # int8 (default)
uv run tools/export_wii.py models/stories15M.bin models/tokenizer.bin sd/llama/ --dtype f32  # exact fp32
```

Copy `sd/llama/` to the root of the SD card, so you have `SD:/llama/model.bin` and `SD:/llama/tokenizer.bin`.

**2. Check the files on your PC first** (optional, catches export problems before you touch the Wii):

```sh
make -C host
./host/iiama2-host sd/llama/model.bin sd/llama/tokenizer.bin -i "Once upon a time"
```

**3. Build for the Wii.** Install devkitPro with the `wii-dev` group (devkitPPC, libogc, libfat) and run `make`, or skip the install and use the official container:

```sh
make                                                              # produces iiama2.dol
docker run --rm -v "$PWD":/src -w /src devkitpro/devkitppc make   # same, no install
```

**4. Run it.** Either:

- **Over Wi-Fi:** open the Homebrew Channel, note its IP, then `export WIILOAD=tcp:<ip>` and `make run`.
- **From the SD card:** copy `iiama2.dol` to `SD:/apps/iiama2/boot.dol` along with `hbc/meta.xml`, and launch it from the Homebrew Channel.

## Controls

| Button | Action |
|---|---|
| LEFT / RIGHT | Pick a prompt |
| UP / DOWN | Temperature (0.0 = greedy, same story every time) |
| + / - | Engine: CPU, split, or GPU (see below) |
| A | Tell a story |
| B | Stop mid-story |
| 1 | Back to the menu (after a story) |
| 2 | Save a screenshot to `SD:/llama/screenshot.bmp` |
| HOME | Back to the Homebrew Channel |

Prompts live in the `PROMPTS` array at the top of `source/main.c`.

To tell a story without touching the Wii Remote, put an `autoplay.txt` in `SD:/llama/`. Optional lines: `engine N` and `prompt N` (0-based, in menu order), `temp X`, and `screenshot` to save the screen as soon as the story ends. The Dolphin screenshot [below](#dolphin) is `engine 2`, `prompt 1`, `temp 0`, `screenshot`.

## The GPU

Each TEV stage computes `d ± lerp(a, b, c)` on 8-bit inputs, with an 11-bit signed register carried from stage to stage. Set it up right and that's a multiply-accumulate:

- **Weights** go in RGBA8 textures as `q + 128`, the same int8 values the CPU uses. Each framebuffer pixel holds three output rows, one per color channel.
- **Activations** go in the four konst colors, one byte per stage. Every row multiplies the same `x`, so it never needs to be a texture.
- **Stage k** is `lerp(HALF, TEXC, KONST)` with a bias that cancels the HALF, so it adds exactly `±round(q * x_k / 256)` to the running sum. Sixteen stages make a 16-column pass per draw. Stage 0 starts the sum from a constant rather than from the PREV register, because a real Wii doesn't reset PREV between pixels.
- **The CPU** copies the framebuffer back and adds up the passes in float, applying the weight and activation scales.

The catch is getting an 11-bit sum out through 8-bit pixels. There are two output modes, and the engine switch picks between them:

| Engine | What runs where |
|---|---|
| CPU | Everything on the CPU, int8 weights |
| split | Transformer layers on the CPU. The GPU computes rough logits for the whole vocab, and the CPU recomputes the top 64 exactly. |
| GPU, fast | Every weight matmul on the TEV. `clamp8`: 16 columns per draw, partial sums saturate past ±128. |
| GPU, exact | `exact11`: 8 columns per draw, plus a second draw that writes the high bits, so sums are exact. About 2.4x the GPU work. |

Attention, norms and SiLU stay on the CPU in every engine.

### Accuracy

stories15M, 1,154 positions of fp32-sampled stories replayed through each engine in the simulator, which the Wii's GPU matches bit for bit (`tools/tev_sim/eval.py`; full report with sample stories in [`tools/tev_sim/results/stories15M.md`](tools/tev_sim/results/stories15M.md)):

| Engine | Top-1 agrees with fp32 | KL vs fp32 (nats) | Perplexity |
|---|---|---|---|
| fp32 CPU | 100% | 0 | 2.77 |
| int8 CPU | 98.4% | 0.0005 | 2.77 |
| split | 98.4% | 0.0008 | 2.77 |
| GPU, fast | 87.3% | 0.061 | 2.97 |
| GPU, exact | 96.9% | 0.0022 | 2.77 |

The split engine matches int8 CPU on top-1 and perplexity: the true best token was inside the GPU's top 64 at all 1,154 positions. GPU-fast stories wander away from the CPU's within a few dozen tokens but stay grammatical.

### Speed on a real Wii

stories15M, int8, the bench's 64-token greedy story (see [below](#running-it-on-a-real-wii)):

| Engine | tok/s | vs CPU |
|---|---|---|
| CPU | 7.41 | 1x |
| split | 12.29 | 1.7x |
| GPU, fast | 20.55 | 2.8x |
| GPU, exact | 9.46 | 1.3x |

Where a GPU-fast token's 49 ms goes:

| | ms |
|---|---|
| CPU issuing GX commands | 4.6 |
| CPU waiting for the GPU | 9.9 |
| CPU reading the framebuffer back and summing passes | 29.0 |
| Everything else (attention, norms, SiLU, sampling) | 5.1 |

The GPU itself is the small part. The readback costs about 13 cycles per output row per pass, four times what its instructions suggest, most likely in cache misses: the framebuffer copy and the weight scales all stream in from main memory. `tools/cost_model.py` was fitted to these numbers and reproduces each engine within 3%, so it can price an idea before anyone writes it.

## Running it on a real Wii

Put an empty `bench.txt` in `SD:/llama/` and the app runs every engine on the same greedy story at boot. It shows tok/s and the GPU time split (issuing, waiting, reading back) on screen, and writes the same to `SD:/llama/bench_out.txt`. A `steps N` line in `bench.txt` sets the story length (default 64). Delete `bench.txt` to get the normal menu back. To check a bench run against the simulator:

```sh
uv run tools/tev_sim/check_dolphin.py models/stories15M.bin models/tokenizer.bin bench_out.txt
```

If the GPU engines disagree with the CPU, put an empty `probe.txt` in `SD:/llama/` as well. The app then runs `source/gx_probe.c` first: small draws that test texture sampling, the konst registers, a color register's value and how out-of-range pixels are written, one multiply-add stage over every weight byte, and the exact11 halving stages, plus raw dumps of real matmuls, each hashed three times. It writes `SD:/llama/probe.bin`, and this says what came out different and how:

```sh
uv run tools/tev_sim/check_probe.py models/stories15M.bin probe.bin
```

That's how the difference below was found. If you run it on a Wii U in Wii mode, an issue with your `bench_out.txt` and probe results would be interesting.

## Memory budget

The Wii has 24 MB MEM1 + 64 MB MEM2. After the system and the app take their share, roughly 70 MB is usable, and no single allocation can span both banks. The loader allocates every matrix separately (small ones first, the big embedding table last) so the weights pack across both.

`export_wii.py` prints an estimate for every model it converts. With the default 256-token context:

| Model | int8 (default) | fp32 |
|---|---|---|
| stories260K | ~1 MB | ~1 MB |
| stories15M | ~20 MB | ~61 MB (very tight, may not fit) |
| stories42M | ~53 MB | won't fit |
| stories110M | ~136 MB, won't fit | won't fit |

The GPU engines need their own copy of the weights as textures, about 1.4 bytes per weight. stories15M takes ~21 MB of textures on top of its ~20 MB, which fits. stories42M doesn't, so it runs CPU only (the app uploads what fits and tells you).

int8 is the default because it's about 4x smaller, and it should be faster too. Its inner loop costs ~3.3 cycles per multiply-add, because Broadway has one load/store unit and int8 needs three loads per multiply-add, versus ~2.1 cycles for fp32. But fp32 moves 4x the bytes, so it only wins if the CPU can stream memory at nearly the bus's full 1.9 GB/s, which an in-order core without prefetching shouldn't manage. Paired-singles dequantization (see below) would cut int8's loads.

The exporter caps context at 256 tokens (`--seq-len`), which is about one TV screen of text. The KV cache grows with context: at its native 1024, stories42M's cache alone would be ~34 MB and the model wouldn't fit.

stories260K uses its own 512-token tokenizer (`tok512.bin`, in the same Hugging Face repo), not the default `tokenizer.bin`. Its group size (4) is too small for the GPU path, so it's CPU only.

## Dolphin

Dolphin boots `iiama2.dol` directly and is the fastest dev loop. For SD access, point Dolphin's virtual SD card at a folder (or image) containing `llama/`. Or let a script do it:

```sh
tools/dolphin_bench.sh sd/llama > dolphin.txt   # headless bench, throwaway Dolphin profile, needs mtools
uv run tools/tev_sim/check_dolphin.py models/stories15M.bin models/tokenizer.bin dolphin.txt
PROBE=1 tools/dolphin_bench.sh sd/llama          # the GPU probes instead; keeps probe.bin
```

The first line builds an SD image, runs the bench mode with the software renderer, and prints the results. The second compares them against the simulator.

![A story about a tired grumpy turtle king, white console text on black](docs/dolphin.png)

<sub>stories15M with every matmul on the GPU (the fast engine), greedy, prompt "The grumpy turtle king". The Wii build in Dolphin, captured by the app's own screenshot feature.</sub>

Four cautions:

- **Speed numbers from Dolphin are meaningless.** Its JIT runs on your desktop CPU, and GPU work finishes instantly in emulated time. Benchmark on the real console.
- **Dolphin 2606 and later won't boot a DOL with unpadded sections**, which devkitPPC's `elf2dol` produces. The Makefile pads them with `tools/pad_dol.py`.
- **Turn off Dolphin's emulated-memory override** (MEM1/MEM2 size) so you don't accidentally depend on RAM a real Wii doesn't have.
- **Don't read a TEV register before writing it.** See below.

### Dolphin vs a real Wii

The first build matched the simulator bit for bit in Dolphin and told garbage stories on a real Wii. The probes found why: **Dolphin starts every pixel's TEV registers from the values set with `GX_SetTevColor`, but a real Wii doesn't reset PREV between pixels.** As far as the probes can tell, each pixel's PREV starts with whatever an earlier pixel left there, so the matmul's starting sum of 128 (or 0) was leftover garbage, and repeated runs of the same matmul gave different results. Everything else the probes test matched Dolphin exactly. The fix is to never read PREV before writing it: stage 0 takes its starting value from a constant input (`GX_CC_HALF` or `GX_CC_ZERO`) instead.

## How it was checked

Output was compared token for token, with greedy decoding and seeded sampling, against upstream llama2.c on randomly initialized checkpoints that exercise GQA, unshared classifiers, and odd group sizes. Comparisons covered:

- **Upstream vs this engine (fp32):** bit-identical text with `-DLLM_EXACT_MATMUL -ffp-contract=off`, on both x86 and big-endian 32-bit PowerPC (`qemu-ppc`, `-mcpu=750`). Also identical on stories15M itself (macOS, arm64).
- **x86 vs PowerPC:** identical text for both fp32 and int8 builds.

The GPU path was checked in layers:

- **The TEV math.** `tools/tev_sim/tev.py` follows the combiner arithmetic in Dolphin's software renderer (`Tev.cpp`). The probes confirmed it on a real Wii: one multiply-add stage for all 256 weight bytes against 10 activation values in both signs, the halving stages, and out-of-range writes, which wrap like a u8 cast.
- **The simulator against Dolphin.** Run in Dolphin, the Wii build's integer pass sums match the simulator bit for bit, for three matrix shapes in both output modes. Over 64 greedy tokens, four of the five engines tell the same story as the simulator. GPU-fast diverges at token 41, at a near-tie (the top two logits differ by 0.02, against a median gap of 1.9), where the CPU-side float rounding of numpy and C differs.
- **The simulator against a real Wii.** The same six integer hashes, identical across repeated runs. Over 64 greedy tokens, the Wii's stories match the simulator's exactly for four of the five engines, and GPU-fast diverges at the same token-41 near-tie as in Dolphin.
- **The simulator against the C engine.** The numpy port in `tools/tev_sim/llama.py` reproduces the C engine's greedy stories exactly, fp32 and int8.

Note that the default build sums each matmul row with 4 accumulators for speed, so its fp32 output can differ from upstream in the last bits. Build with `-DLLM_EXACT_MATMUL` if you want upstream-exact numerics.

## What's in here

| Path | What it is |
|---|---|
| `source/llm.c`, `llm.h` | The inference engine (forward pass, BPE tokenizer, sampler). Portable C, no mmap, no `exit()`. |
| `source/gx_matmul.c` | Weight matmuls on the GPU's TEV, for the split and GPU engines. |
| `source/gx_probe.c` | Hardware probes for the GPU path (`probe.txt`). |
| `source/main.c` | The Wii app: SD loading with a progress bar, prompt menu, engine switch, word-wrapped output, tok/s, bench mode, autoplay, screenshots. |
| `Makefile` | devkitPPC build (stock devkitPro Wii template + libfat). |
| `host/` | Desktop build of the same engine, for testing exported files on your PC or under `qemu-ppc`. |
| `hbc/meta.xml` | Homebrew Channel listing. |
| `tools/export_wii.py` | Converts a llama2.c checkpoint + tokenizer to big-endian files, optionally int8. Needs only numpy. |
| `tools/tev_sim/` | Bit-exact numpy simulator of the TEV math, the accuracy evaluation, and the bench and probe checkers. |
| `tools/cost_model.py` | Costs each engine on a real Wii, fitted to one. |
| `tools/dolphin_bench.sh` | Runs the bench mode headless in Dolphin. |
| `tools/pad_dol.py` | Pads DOL sections so newer Dolphin will boot them. |
| `docs/` | The photo and the Dolphin screenshot. |

## Ideas for next steps

- **Overlap the CPU and GPU.** Right now the CPU waits for each batch of draws and then sums it. Double-buffering the framebuffer copy would let the CPU sum batch k while the GPU draws batch k+1. Splitting a matmul's rows between the two would put both to work.
- **Shrink the readback.** It's 60% of a GPU-fast token. The most promising fix is to let the framebuffer do the summing: blend every pass into the same pixels, alternating add and subtract blend modes so the 128 biases cancel, and read back one byte per row instead of one per pass. That needs per-row weight scales and one activation scale per token. In the simulator the blended classifier still had the true best token in its top 64 at all 240 positions tested, and the fitted cost model says it takes split from 12.3 to ~15.6 tok/s and GPU-fast (with a CPU top-64 rescore) from 20.4 to ~31.
- **Cheaper GX calls.** libogc's TEV setters do read-modify-write on shadow state; about 3,600 instructions per pass could be a few hundred as raw register writes.
- **Paired-singles dequantization.** Broadway's `psq_l` instruction loads two int8 values and converts them to floats in hardware, scaled by a GQR register. Nintendo built it for vertex compression, and it's a near-perfect fit for int8 weights. Swapping the lookup table in `matmul_q8` for `psq_l` + `ps_madd` would cut the CPU engine's loads by more than half. Adding `dcbt` prefetches would hide some of the memory stalls.
- **Train your own.** A ~15M model trained on a narrow domain (a grumpy turtle king, say) on a single gaming GPU is more coherent and much funnier than a general one. Use a small vocab while you're at it: on stories15M, the 32000-token output layer is ~60% of the work per token, and a 4096-token vocab (llama2.c's `--vocab_source=custom`) roughly halves it. Export with llama2.c's `export.py --version 0`, then run it through `export_wii.py`.

## Credits

- [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy, which the engine is ported from, and the [tinyllamas](https://huggingface.co/karpathy/tinyllamas) checkpoints.
- [TinyStories](https://arxiv.org/abs/2305.07759) by Ronen Eldan and Yuanzhi Li, the dataset those models were trained on.
- [devkitPro](https://devkitpro.org/) for devkitPPC, libogc and libfat.
- [Dolphin](https://dolphin-emu.org/), whose software renderer is the reference for the TEV math, and which ran every line of this before any real Wii did.

## License

MIT, same as llama2.c. See [LICENSE](LICENSE).

Not affiliated with or endorsed by Nintendo. Wii is a trademark of Nintendo.
