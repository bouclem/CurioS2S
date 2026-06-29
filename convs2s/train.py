import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import ConvS2S
from .data import (
    SyntheticSeqDataset,
    get_dataloader,
    make_vocab,
    VOCAB_SIZE,
    PAD_IDX,
    BOS_IDX,
    EOS_IDX,
    encode,
    decode,
)


def train_model(
    epochs: int = 20,
    batch_size: int = 32,
    dim: int = 128,
    num_layers: int = 4,
    kernel_size: int = 3,
    lr: float = 3e-4,
    device: str = "cpu",
    plot_dir: str = "plots",
) -> dict:
    """Train ConvS2S on the synthetic reversal task.

    Returns a dict with training history and saves a loss plot via matplotlib.
    """
    os.makedirs(plot_dir, exist_ok=True)

    dataset = SyntheticSeqDataset(num_samples=2000)
    dataloader = get_dataloader(dataset, batch_size=batch_size)

    model = ConvS2S(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        dim=dim,
        num_layers=num_layers,
        kernel_size=kernel_size,
        dropout=0.1,
        padding_idx=PAD_IDX,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    history = {"epoch": [], "loss": [], "accuracy": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            # Decoder input: tgt without last token (shift right)
            # Target: tgt without first token (BOS)
            dec_input = tgt[:, :-1]
            dec_target = tgt[:, 1:]

            logits = model(src, dec_input)  # (batch, tgt_len-1, vocab_size)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                dec_target.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = dec_target.ne(PAD_IDX)
            correct = (preds == dec_target) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / max(total_tokens, 1)
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)

        print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

    # --- Matplotlib: loss & accuracy plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["epoch"], history["loss"], "b-o", markersize=3, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("ConvS2S Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["epoch"], history["accuracy"], "r-o", markersize=3, label="Token Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("ConvS2S Token Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "convs2s_training.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nTraining plot saved to: {plot_path}")

    # --- Quick inference demo ---
    model.eval()
    print("\n--- Inference Demo ---")
    test_seqs = [["3", "5", "7"], ["9", "4", "6", "2"], ["8", "1"]]
    for seq in test_seqs:
        src_tensor = torch.tensor([encode(seq)], dtype=torch.long).to(device)
        output = model.generate(src_tensor, max_len=20, bos_idx=BOS_IDX, eos_idx=EOS_IDX)
        out_tokens = decode(output[0].cpu().tolist())
        print(f"  Input:    {seq}")
        print(f"  Output:   {out_tokens}")
        print(f"  Expected: {list(reversed(seq))}")
        print()

    return {"model": model, "history": history}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    train_model(device=device)
