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


# Handmade dataset: text facts + math problems (not stories)
SAMPLES = [
    # Math: addition
    ("what is 2+2", "4"),
    ("what is 3+5", "8"),
    ("what is 7+1", "8"),
    ("what is 9+3", "12"),
    ("what is 4+6", "10"),
    ("what is 5+5", "10"),
    # Math: subtraction
    ("what is 4-2", "2"),
    ("what is 8-3", "5"),
    ("what is 10-7", "3"),
    ("what is 6-4", "2"),
    ("what is 9-1", "8"),
    # Math: multiplication
    ("what is 3*4", "12"),
    ("what is 5*2", "10"),
    ("what is 7*3", "21"),
    ("what is 6*6", "36"),
    ("what is 2*8", "16"),
    # Math: division
    ("what is 8/2", "4"),
    ("what is 9/3", "3"),
    ("what is 10/5", "2"),
    ("what is 12/4", "3"),
    # Text: facts (not stories)
    ("color of sky", "blue"),
    ("color of grass", "green"),
    ("color of sun", "yellow"),
    ("color of blood", "red"),
    ("color of coal", "black"),
    ("color of snow", "white"),
    ("capital of france", "paris"),
    ("capital of japan", "tokyo"),
    ("capital of italy", "rome"),
    ("capital of egypt", "cairo"),
    ("animal that barks", "dog"),
    ("animal that meows", "cat"),
    ("animal that moos", "cow"),
    ("opposite of hot", "cold"),
    ("opposite of big", "small"),
    ("opposite of fast", "slow"),
    ("opposite of light", "dark"),
    ("opposite of up", "down"),
    ("opposite of good", "bad"),
]


class CurioDataset(Dataset):
    """Handmade dataset with text facts and math problems.

    Not stories — just direct Q&A pairs that test the model's
    ability to learn factual knowledge and arithmetic.
    """

    def __init__(self, samples: list = None, repeat: int = 10):
        if samples is None:
            samples = SAMPLES
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
