import torch
import torch.nn as nn
from collections import Counter


class CharTokenizer:
    """Character-level tokenizer for CurioNet.

    Builds a vocabulary from raw text. Supports encode/decode
    with special tokens (pad, bos, eos).

    NOT a subword tokenizer (no BPE/SentencePiece).
    NOT a word tokenizer.
    Character-level — every character is a token.
    """

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(self):
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        self.chars = []
        self.vocab = list(self.special_tokens)
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

    def build(self, text: str):
        """Build vocabulary from raw text."""
        counter = Counter(text)
        self.chars = sorted(counter.keys())
        self.vocab = list(self.special_tokens) + self.chars
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        return self

    def encode(self, text: str, add_special: bool = True) -> list:
        ids = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        if add_special:
            ids = [self.bos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, ids: list) -> str:
        chars = []
        for i in ids:
            if i == self.eos_idx:
                break
            if i in (self.pad_idx, self.bos_idx):
                continue
            chars.append(self.idx_to_char.get(i, "?"))
        return "".join(chars)

    def save(self, path: str):
        """Save tokenizer vocab to file."""
        with open(path, "w", encoding="utf-8") as f:
            for token in self.vocab:
                if token == "\n":
                    f.write("\\n\n")
                elif token == "\t":
                    f.write("\\t\n")
                else:
                    f.write(f"{token}\n")

    def load(self, path: str):
        """Load tokenizer vocab from file."""
        self.vocab = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                token = line.rstrip("\n")
                if token == "\\n":
                    token = "\n"
                elif token == "\\t":
                    token = "\t"
                self.vocab.append(token)
        self.special_tokens = self.vocab[:3]
        self.chars = self.vocab[3:]
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        return self

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
