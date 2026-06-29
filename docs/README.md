# CurioS2S

Convolutional Sequence-to-Sequence model with a **Curiosity Drive** — a generative mechanism that creates wonder states, making the model think deeper and explore further.

Based on ConvS2S (Gehring et al., 2017) with an added curiosity module that is **NOT attention, NOT a detector, NOT a score** — it is a generative drive.

## Architecture

```
CurioS2S
├── ConvEncoder
│   ├── PositionalEmbedding (token + position)
│   ├── ConvBlock x N (Conv1d → GLU → residual)
│   └── CuriosityDrive (wonder generation → thinking → insights)
│
├── ConvDecoder
│   ├── PositionalEmbedding (token + position)
│   ├── ConvBlock x N (causal Conv1d → GLU → residual)
│   ├── ConvAttention x N (decoder ↔ encoder)
│   └── CuriosityDrive (wonder generation → thinking → insights)
│
└── Output projection → vocab logits
```

### Curiosity Drive

Curiosity is a generative drive. It does NOT weight inputs (attention), NOT classify patterns (detector), NOT assign values (score).

What it does:
1. **Generate wonder** — new internal states from current representation
2. **Process wonder** — dedicated conv layers let the model "think"
3. **Generate insights** — from wonder processing
4. **Integrate insights** — enriches the representation

## Project Structure

```
Net/
├── data/
│   ├── math.txt              # 80 math Q&A pairs (add, sub, mul, div)
│   └── talk.txt              # 68 text fact Q&A pairs (colors, capitals, etc.)
├── curios2s/
│   ├── __init__.py           # Package exports
│   ├── model.py              # CurioS2S, ConvEncoder, ConvDecoder, ConvAttention
│   ├── curiosity.py          # CuriosityDrive, WonderGenerator, WonderConv
│   ├── data.py               # Char-level vocab, CurioDataset, file loading
│   ├── train.py              # Training loop + matplotlib plots + weight saving
│   └── chat.py               # Interactive chat with saved weights
├── checkpoints/              # Saved model weights (gitignored)
├── plots/                    # Training plots (gitignored)
├── docs/
│   ├── README.md             # This file
│   ├── TODO.md               # Pending tasks
│   └── CHANGELOG.md          # Version history
├── .gitignore
└── requirements.txt
```

## Data Format

Each line in `data/*.txt`:

```
question|answer
```

Examples:
```
what is 2+2|4
color of sky|blue
capital of japan|tokyo
```

## Usage

### Train

```bash
pip install -r requirements.txt
python -m curios2s.train
```

Saves:
- `plots/curios2s_training.png` — loss & accuracy curves
- `checkpoints/curios2s.pt` — model weights + config

### Chat

```bash
python -m curios2s.chat
```

Interactive prompt:
```
You> what is 2+2
CurioS2S> 4  [OK, expected: 4]

You> color of sky
CurioS2S> blue  [OK, expected: blue]
```

### Programmatic API

```python
from curios2s import CurioS2S, chat, ask, load_model

model = load_model()
answer = ask(model, "what is 3*4")
print(answer)  # "12"
```

## Dependencies

- `torch>=2.0.0`
- `matplotlib>=3.7.0`

## Key Design Decisions

- **Char-level vocabulary** — a-z, 0-9, `+-*/= ?` + special tokens
- **Curiosity in both encoder and decoder** — wonder is generated after conv processing
- **Weight checkpointing** — full model config saved with weights for easy reloading
- **File-based datasets** — `data/talk.txt` and `data/math.txt`, easy to extend
