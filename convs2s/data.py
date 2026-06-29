import torch
from torch.utils.data import Dataset, DataLoader

# Special tokens
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2

# Simple vocab: digits 3-12 as "words"
VOCAB = ["<pad>", "<bos>", "<eos>"] + [str(i) for i in range(10)]
VOCAB_SIZE = len(VOCAB)
TOKEN_TO_IDX = {tok: i for i, tok in enumerate(VOCAB)}
IDX_TO_TOKEN = {i: tok for i, tok in enumerate(VOCAB)}


def make_vocab() -> dict:
    return {"vocab": VOCAB, "size": VOCAB_SIZE, "tok2idx": TOKEN_TO_IDX, "idx2tok": IDX_TO_TOKEN}


def encode(seq: list, add_special: bool = True) -> list:
    ids = [TOKEN_TO_IDX[tok] for tok in seq]
    if add_special:
        ids = [BOS_IDX] + ids + [EOS_IDX]
    return ids


def decode(ids: list) -> list:
    tokens = []
    for i in ids:
        if i == EOS_IDX:
            break
        if i in (PAD_IDX, BOS_IDX):
            continue
        tokens.append(IDX_TO_TOKEN[i])
    return tokens


class SyntheticSeqDataset(Dataset):
    """Synthetic sequence reversal task: learn to reverse the input sequence.

    E.g. input:  [3, 5, 7] -> target: [7, 5, 3]
    This is a simple but non-trivial task that tests the model's
    ability to capture order-dependent mappings.
    """

    def __init__(self, num_samples: int = 2000, min_len: int = 3, max_len: int = 8):
        self.samples = []
        for _ in range(num_samples):
            seq_len = torch.randint(min_len, max_len + 1, (1,)).item()
            src_seq = [str(torch.randint(0, 10, (1,)).item()) for _ in range(seq_len)]
            tgt_seq = list(reversed(src_seq))
            self.samples.append((encode(src_seq), encode(tgt_seq)))

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


def get_dataloader(dataset: SyntheticSeqDataset, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b),
    )
