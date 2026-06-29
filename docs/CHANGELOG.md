# Changelog

## v0.2.0 — CurioS2S + Curiosity Drive

### Added
- `curiosity.py` — CuriosityDrive, WonderGenerator, WonderConv
  - Generative wonder states (NOT attention, NOT detector, NOT score)
  - Dedicated "thinking" conv layers for processing wonder
  - Insight generation and integration
- `data/talk.txt` — 68 text fact Q&A pairs (colors, capitals, animals, opposites, planets, numbers)
- `data/math.txt` — 80 math Q&A pairs (addition, subtraction, multiplication, division)
- `chat.py` — interactive chat with saved weights
  - `load_model()`, `ask()`, `chat()` functions
  - Shows [OK]/[MISS] for known questions
- Weight checkpointing — saves to `checkpoints/curios2s.pt` with model config + history
- `docs/` folder with README.md, TODO.md, CHANGELOG.md

### Changed
- Renamed `ConvS2S` → `CurioS2S` (ConvS2S kept as backward-compatible alias)
- Renamed package folder `convs2s/` → `curios2s/`
- Char-level vocabulary (a-z, 0-9, operators, space, ?) replacing word-level
- `data.py` loads from `data/*.txt` files instead of hardcoded list
- `CurioDataset` replaces `SyntheticSeqDataset`
- Training: 30 epochs, batch_size=16, 148 samples (up from 38)
- Inference demo uses text + math questions instead of sequence reversal

### Fixed
- KeyError: vocab generation produced tokens not in vocabulary
- Dimension mismatch in ConvAttention combine layer (3*dim not 2*dim)

---

## v0.1.0 — Initial ConvS2S

### Added
- `model.py` — ConvS2S, ConvEncoder, ConvDecoder, ConvAttention, ConvBlock, GLU
  - Based on Gehring et al. (2017): ConvS2S
  - Positional embeddings, gated linear units, residual connections
  - Scaled dot-product attention with encoder embedding + conv output
- `data.py` — SyntheticSeqDataset (sequence reversal task)
- `train.py` — training loop with matplotlib loss/accuracy visualization
- `.gitignore`, `requirements.txt`
