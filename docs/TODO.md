# TODO

## Model
- [ ] Add beam search decoding (currently greedy only)
- [ ] Add temperature/sampling for more diverse generation
- [ ] Experiment with higher `curiosity_budget` (more wonder cycles)
- [x] Add learning rate scheduler (cosine or warmup)
- [ ] Add validation split for proper evaluation

## Data
- [ ] Add more diverse math (two-digit operations, parentheses)
- [ ] Add more text categories (science, geography, history)
- [ ] Add data augmentation (paraphrased questions)
- [ ] Support loading custom .txt files at runtime

## Training
- [x] Add early stopping when loss plateaus
- [ ] Add gradient accumulation for larger effective batch size
- [ ] Log curiosity wonder norms for analysis
- [x] Save best model (by validation accuracy) instead of last

## Chat
- [ ] Add conversation history (multi-turn context)
- [x] Add streaming output (char-by-char)
- [ ] Add confidence display
- [ ] Add fallback for unknown questions

## Docs
- [x] Add architecture diagram (visual)
- [ ] Add benchmark results table
- [ ] Add comparison: ConvS2S vs CurioS2S (with/without curiosity)
