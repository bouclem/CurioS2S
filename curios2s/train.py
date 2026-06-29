import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import CurioS2S
from .data import (
    CurioDataset,
    get_dataloader,
    make_vocab,
    VOCAB_SIZE,
    PAD_IDX,
    BOS_IDX,
    EOS_IDX,
    encode,
    decode,
    SAMPLES,
)


def _save_checkpoint(model, config, history, path):
    """Save a checkpoint to the given path."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "history": history,
    }, path)


class WarmupScheduler:
    """Linear warmup then constant LR.

    LR ramps from 0 to `lr` over `warmup_epochs` epochs,
    then stays at `lr` for the rest of training.
    """

    def __init__(self, optimizer, lr: float, warmup_epochs: int = 5):
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def step(self, epoch: int):
        self.current_epoch = epoch
        if epoch <= self.warmup_epochs:
            scale = epoch / self.warmup_epochs
        else:
            scale = 1.0
        current_lr = self.lr * scale
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr
        return current_lr


def train_model(
    epochs: int = 50,
    batch_size: int = 16,
    dim: int = 128,
    num_layers: int = 4,
    kernel_size: int = 3,
    lr: float = 3e-4,
    device: str = "cpu",
    plot_dir: str = "plots",
    checkpoint_dir: str = "checkpoints",
    patience: int = 4,
    warmup_epochs: int = 5,
) -> dict:
    """Train CurioS2S on handmade text + math dataset.

    Returns a dict with training history and saves a loss plot via matplotlib.
    Saves best.pt (lowest loss) and latest.pt (last epoch) to checkpoint_dir.
    Stops early if loss doesn't improve for `patience` consecutive epochs.
    """
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_config = {
        "src_vocab_size": VOCAB_SIZE,
        "tgt_vocab_size": VOCAB_SIZE,
        "dim": dim,
        "num_layers": num_layers,
        "kernel_size": kernel_size,
        "dropout": 0.1,
        "padding_idx": PAD_IDX,
    }

    dataset = CurioDataset(repeat=20)
    dataloader = get_dataloader(dataset, batch_size=batch_size)

    model = CurioS2S(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        dim=dim,
        num_layers=num_layers,
        kernel_size=kernel_size,
        dropout=0.1,
        padding_idx=PAD_IDX,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = WarmupScheduler(optimizer, lr=lr, warmup_epochs=warmup_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    history = {"epoch": [], "loss": [], "accuracy": [], "ppl": [], "lr": []}
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        current_lr = scheduler.step(epoch)
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            dec_input = tgt[:, :-1]
            dec_target = tgt[:, 1:]

            logits = model(src, dec_input)

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
        ppl = math.exp(min(avg_loss, 20))  # clamp to avoid overflow
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)
        history["ppl"].append(ppl)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch:3d}/{epochs} | LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | Acc: {accuracy:.4f}")

        # --- Save latest.pt every epoch ---
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        _save_checkpoint(model, model_config, history, latest_path)

        # --- Save best.pt if loss improved ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_path = os.path.join(checkpoint_dir, "best.pt")
            _save_checkpoint(model, model_config, history, best_path)
            print(f"  -> New best model saved (loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{patience})")

        # --- Early stopping ---
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # --- Matplotlib: loss, PPL & accuracy plot ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].plot(history["epoch"], history["loss"], "b-o", markersize=3, label="Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("CurioS2S Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["epoch"], history["ppl"], "g-o", markersize=3, label="Perplexity")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PPL")
    axes[1].set_title("CurioS2S Perplexity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["epoch"], history["accuracy"], "r-o", markersize=3, label="Token Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("CurioS2S Token Accuracy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "curios2s_training.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nTraining plot saved to: {plot_path}")

    # --- Quick inference demo ---
    model.eval()
    print("\n--- Inference Demo ---")
    test_questions = [
        "what is 2+2",
        "what is 3*4",
        "color of sky",
        "capital of japan",
        "opposite of hot",
        "animal that barks",
    ]
    for q in test_questions:
        src_tensor = torch.tensor([encode(q)], dtype=torch.long).to(device)
        output = model.generate(src_tensor, max_len=20, bos_idx=BOS_IDX, eos_idx=EOS_IDX)
        answer = decode(output[0].cpu().tolist())
        expected = ""
        for sq, sa in SAMPLES:
            if sq == q:
                expected = sa
                break
        print(f"  Q: {q}")
        print(f"  A: {answer}")
        if expected:
            print(f"  Expected: {expected}")
        print()

    print(f"Best model saved to: {os.path.join(checkpoint_dir, 'best.pt')}")
    print(f"Latest model saved to: {os.path.join(checkpoint_dir, 'latest.pt')}")

    return {"model": model, "history": history}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    train_model(device=device)
