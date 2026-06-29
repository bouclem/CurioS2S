import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from .tokenizer import CharTokenizer

# --- Tokenizer (built from training data) ---
_tokenizer = None

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2

# --- WikiText-2 cache ---
_CACHE_DIR = os.path.join("data", "wikitext2")


def _download_wikitext2():
    """Download WikiText-2 raw via HuggingFace datasets library."""
    print("Downloading WikiText-2 via datasets library...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", cache_dir=_CACHE_DIR)
    print("Download complete.")
    return ds


def _load_raw_text(split: str) -> str:
    """Load raw text from WikiText-2 via datasets library."""
    ds = _download_wikitext2()
    split_map = {"train": "train", "validation": "validation", "test": "test"}
    texts = ds[split_map.get(split, "train")]["text"]
    return "\n".join(t for t in texts if t.strip())


def get_tokenizer() -> CharTokenizer:
    """Get or build the tokenizer from WikiText-2 training data."""
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer

    vocab_path = os.path.join("data", "tokenizer.txt")
    tok = CharTokenizer()

    if os.path.exists(vocab_path):
        tok.load(vocab_path)
    else:
        text = _load_raw_text("train")
        tok.build(text)
        os.makedirs("data", exist_ok=True)
        tok.save(vocab_path)

    _tokenizer = tok
    return tok


# Build tokenizer eagerly so VOCAB_SIZE etc. are available
_tok = get_tokenizer()
VOCAB_SIZE = _tok.vocab_size
CHAR_TO_IDX = _tok.char_to_idx
IDX_TO_CHAR = _tok.idx_to_char


def make_vocab() -> dict:
    return {"vocab": _tok.vocab, "size": VOCAB_SIZE, "tok2idx": CHAR_TO_IDX, "idx2tok": IDX_TO_CHAR}


def encode(text: str, add_special: bool = True) -> list:
    return _tok.encode(text, add_special=add_special)


def decode(ids: list) -> str:
    return _tok.decode(ids)


def _build_corpus(split: str, max_chars: int = 500000) -> str:
    """Load raw text from WikiText-2 and truncate to max_chars."""
    text = _load_raw_text(split)
    return text[:max_chars]


class WikiText2Dataset(Dataset):
    """WikiText-2 character-level continuation dataset.

    Splits text into fixed-length chunks. Input = first half,
    target = second half (text continuation task).

    Args:
        split: 'train', 'validation', or 'test'.
        seq_len: Total sequence length (input + target).
        max_chars: Max chars to use from the corpus.
    """

    def __init__(self, split: str = "train", seq_len: int = 64, max_chars: int = 500000):
        corpus = _build_corpus(split, max_chars=max_chars)

        half = seq_len // 2
        self.samples = []

        for i in range(0, len(corpus) - seq_len, half):
            chunk = corpus[i:i + seq_len]
            if len(chunk) < seq_len:
                break
            src_text = chunk[:half]
            tgt_text = chunk[half:]
            src_ids = encode(src_text)
            tgt_ids = encode(tgt_text)
            self.samples.append((src_ids, tgt_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        src_ids, tgt_ids = self.samples[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch: list, pad_idx: int = PAD_IDX) -> tuple:
    src_seqs, tgt_seqs = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


def get_dataloader(dataset, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda b: collate_fn(b))
