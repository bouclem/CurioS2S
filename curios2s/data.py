import os
import torch
from torch.utils.data import Dataset, DataLoader

# Special tokens
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2

# Character-level vocabulary for text + math
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]
CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789+-*/= ?")
VOCAB = SPECIAL_TOKENS + CHARS
VOCAB_SIZE = len(VOCAB)
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}


def make_vocab() -> dict:
    return {"vocab": VOCAB, "size": VOCAB_SIZE, "tok2idx": CHAR_TO_IDX, "idx2tok": IDX_TO_CHAR}


def encode(text: str, add_special: bool = True) -> list:
    ids = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]
    if add_special:
        ids = [BOS_IDX] + ids + [EOS_IDX]
    return ids


def decode(ids: list) -> str:
    chars = []
    for i in ids:
        if i == EOS_IDX:
            break
        if i in (PAD_IDX, BOS_IDX):
            continue
        chars.append(IDX_TO_CHAR.get(i, "?"))
    return "".join(chars)


# Path to data files
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load_data_file(filename: str) -> list:
    """Load Q|A pairs from a data file. Each line: question|answer"""
    filepath = os.path.join(_DATA_DIR, filename)
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            q, a = line.split("|", 1)
            pairs.append((q.strip(), a.strip()))
    return pairs


def load_samples() -> list:
    """Load all samples from data/talk.txt and data/math.txt."""
    return _load_data_file("talk.txt") + _load_data_file("math.txt")


# Loaded at import time for convenience
SAMPLES = load_samples()


class CurioDataset(Dataset):
    """Handmade dataset with text facts and math problems.

    Not stories — just direct Q&A pairs that test the model's
    ability to learn factual knowledge and arithmetic.
    """

    def __init__(self, samples: list = None, repeat: int = 10):
        if samples is None:
            samples = load_samples()
        self.samples = []
        for src_text, tgt_text in samples:
            self.samples.append((encode(src_text), encode(tgt_text)))
        # Repeat for more training data
        original = list(self.samples)
        for _ in range(repeat - 1):
            self.samples.extend([(list(s), list(t)) for s, t in original])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        src_ids, tgt_ids = self.samples[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch: list, pad_idx: int = PAD_IDX) -> tuple:
    """Pad sequences to the same length within a batch."""
    src_seqs, tgt_seqs = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


def get_dataloader(dataset: CurioDataset, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b),
    )
