# TODO

## Model
- [ ] Add beam search decoding (currently greedy only)
- [ ] Add temperature/sampling for more diverse generation
- [ ] Experiment with higher `curiosity_budget` (more wonder cycles)
- [x] Add learning rate scheduler (warmup)
- [x] Add validation split for proper evaluation
- [ ] Experiment with different `attn_every` values (2, 4, 5)
- [ ] Try removing tiny attention entirely (pure curiosity)

## Data
- [x] Use WikiText-2 dataset (via HuggingFace datasets library)
- [x] Character-level tokenizer (CharTokenizer)
- [ ] Try WikiText-103 for larger-scale comparison
- [ ] Add subword tokenizer option (BPE/SentencePiece)
- [ ] Support loading custom datasets at runtime

## Training
- [x] Add early stopping when loss plateaus
- [ ] Add gradient accumulation for larger effective batch size
- [ ] Log curiosity wonder norms for analysis
- [x] Save best model and latest model
- [ ] Add LR decay after warmup (cosine)
- [x] Train on GPU (CUDA required)
- [x] ~300K parameter budget for both models

## Chat
- [ ] Add conversation history (multi-turn context)
- [x] Add streaming output (char-by-char)
- [ ] Add confidence display
- [ ] Add temperature control in chat

## Comparison
- [x] Create compare.py — benchmark CurioNet vs Transformer on WikiText-2
- [x] Track test loss, test PPL, inference time
- [x] Generate comparison plots (train/val loss, PPL, params, time)
- [x] Sample generation comparison
- [ ] Add scaling comparison (vary dim, layers, data size)
- [ ] Add memory usage comparison
- [ ] Add generalization test (unseen text)

## Docs
- [x] Add architecture diagram (SVG)
- [x] Add benchmark instructions (compare.py)
- [x] Add comparison: CurioNet vs Transformer
- [x] Document WikiText-2 dataset and tokenizer
- [x] Add config.yaml documentation
- [ ] Add benchmark results table (after running)
