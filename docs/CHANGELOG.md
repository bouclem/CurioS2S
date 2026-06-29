# Changelog

## v1.2.0 — WonderMixer Architecture (Global Curiosity)

### Changed
- Replaced `WonderConv` (local convolutions) with **`WonderMixer`** (global curiosity-driven token mixing)
- CurioNet now has a **global receptive field** like Transformer — no more local-only convolutions
- CurioNet param budget adjusted: `dim=40, num_heads=4` → ~293K params (was dim=46 with convs)
- `config.yaml` updated: removed `num_wonder_convs` and `kernel_size`, added `num_heads`
- `compare.py` now **skips training if checkpoints exist** — loads from disk and goes straight to evaluation
- `chat.py` `load_model` reads `num_heads` from checkpoint config
- Architecture SVG diagram updated to show WonderMixer instead of WonderConv

### Architecture: WonderMixer vs Attention
- **Input**: Operates on generated wonder states (not raw embeddings like attention)
- **Affinity**: Uses sigmoid (independent gating) instead of softmax (competitive budget)
- **Mixing**: Each token independently decides what to take — no fixed attention budget
- **Multi-head**: Yes — multiple "curiosity aspects" (like multi-head attention)
- **Causal masking**: Supported for decoder (left-only, prevents seeing future tokens)

### Added
- `WonderMixer` class in `curiosity.py` — global token mixing via curiosity affinity
- `min_len` parameter in `generate()` — prevents premature eos prediction during autoregressive generation
- Checkpoint loading in `compare.py` — `_load_checkpoint()` function

### Removed
- `WonderConv` class (replaced by `WonderMixer`)
- `num_wonder_convs` and `kernel_size` parameters from all model configs
- Causal conv padding workaround (no longer needed — WonderMixer handles causality natively)

---

## v1.1.0 — WikiText-2 Benchmark

### Changed
- Replaced custom Q&A dataset with **WikiText-2** (raw) via HuggingFace `datasets` library
- Dataset downloads automatically on first run, cached in `data/wikitext2/`
- Both models configured for **~300K parameters** (CurioNet: dim=46, 2 layers; Transformer: dim=64, 2 layers, 4 heads)
- Training now runs on **GPU** (CUDA required)
- `train.py` renamed from `train_and_compare()` to `train_curionet()` — trains CurioNet only, saves `curionet_best.pt`/`curionet_latest.pt`
- `chat.py` updated for text continuation (no more Q&A format)

### Added
- `curionet/compare.py` — `compare()` function: trains both models on WikiText-2, benchmarks on test set (loss, PPL, inference time), generates comparison plots, sample generation, declares winner
- `curionet/tokenizer.py` — `CharTokenizer` class: builds char-level vocab from training data, supports encode/decode/save/load
- `config.yaml` — centralized configuration for dataset, tokenizer, model params, training params, device, paths
- Validation split evaluation (train/val loss + PPL tracking)
- Test set evaluation in `compare.py`
- `plots/wikitext2_benchmark.png` — 6-panel comparison plot
- `plots/curionet_wikitext2.png` — CurioNet training plot

### Removed
- `CurioDataset`, `load_samples()`, `data/question.txt`, `data/math.txt` (replaced by WikiText-2)
- `train_and_compare()` (split into `train_curionet()` and `compare()`)
- Manual `urllib`/`zipfile` download (replaced by `datasets` library)

---

## v1.0.0 — CurioNet Architecture

### Architecture Redesign
CurioNet is a full architecture redesign based on curiosity as the primary mechanism, replacing attention. This is not a modification of the old ConvS2S — it's a new architecture from scratch.

### Added
- `curionet/curiosity.py` — Enhanced curiosity components:
  - `WonderGenerator` — generates wonder states (expand→GELU→contract→gate)
  - `WonderConv` — convolutional "thinking" layers for processing wonder
  - `InsightExtractor` — synthesizes insights from processed wonder
  - `CuriosityLayer` — full curiosity cycle (wonder→process→insight→integrate), replaces attention layers
  - `CuriosityBlock` — curiosity layer + FFN, replaces Transformer blocks

- `curionet/model.py` — CurioNet architecture:
  - `CurioNetEncoder` — curiosity blocks with sparse TinyAttention every N layers
  - `CurioNetDecoder` — curiosity blocks + minimal single-head cross-attention
  - `TinyAttention` — minimal single-head attention (NOT primary mechanism, just a small residual)
  - `CurioNet` — full seq2seq model

- `curionet/transformer.py` — `TransformerSeq2Seq` for fair comparison
  - Same parameter budget as CurioNet
  - Standard multi-head attention in every layer

- `curionet/chat.py` — interactive chat with streaming output
- `curionet/data.py` — dataset loading
- `docs/architecture.svg` — SVG architecture diagram
- `docs/README.md` — full documentation with CurioNet vs Transformer comparison

### Key Design
- Curiosity layers are the PRIMARY mechanism (replaces multi-head attention)
- TinyAttention appears every 3 layers (single-head, minimal) for basic info sharing
- Cross-attention is single-head (minimal, only for encoder-decoder flow)
- Curiosity generates NEW states rather than re-weighting existing ones

### Moved
- All previous CurioS2S code moved to `_OLD/`

---

## v0.2.0 — CurioS2S + Curiosity Drive (archived in _OLD/)

### Added
- CuriosityDrive integrated into ConvS2S encoder/decoder
- Char-level vocabulary, file-based datasets
- Weight checkpointing (best.pt, latest.pt)
- Early stopping, PPL, warmup LR scheduler
- Interactive chat with streaming output
- Architecture SVG diagram

---

## v0.1.0 — Initial ConvS2S (archived in _OLD/)

### Added
- ConvS2S implementation (Gehring et al., 2017)
- SyntheticSeqDataset (sequence reversal task)
- Training loop with matplotlib visualization
