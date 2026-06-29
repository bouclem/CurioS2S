import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import CurioNet
from .transformer import TransformerSeq2Seq
from .data import (
    WikiText2Dataset,
    get_dataloader,
    VOCAB_SIZE,
    PAD_IDX,
    BOS_IDX,
    EOS_IDX,
    encode,
    decode,
)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class WarmupScheduler:
    def __init__(self, optimizer, lr: float, warmup_epochs: int = 5):
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_epochs = warmup_epochs

    def step(self, epoch: int):
        if epoch <= self.warmup_epochs:
            scale = epoch / self.warmup_epochs
        else:
            scale = 1.0
        current_lr = self.lr * scale
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr
        return current_lr


def _save_checkpoint(model, config, history, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "history": history,
    }, path)


def _load_checkpoint(model, path, device):
    """Load weights into model from checkpoint. Returns history dict or empty dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt.get("history", {"epoch": [], "train_loss": [], "train_ppl": [], "val_loss": [], "val_ppl": [], "lr": []})


def _train_one(
    model: nn.Module,
    name: str,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device: str,
    patience: int,
    warmup_epochs: int,
    checkpoint_dir: str = "checkpoints",
    model_config: dict = None,
) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = WarmupScheduler(optimizer, lr=lr, warmup_epochs=warmup_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    history = {"epoch": [], "train_loss": [], "train_ppl": [], "val_loss": [], "val_ppl": [], "lr": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        current_lr = scheduler.step(epoch)
        model.train()
        total_loss = 0.0
        total_tokens = 0
        epoch_start = time.time()

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            dec_input = tgt[:, :-1]
            dec_target = tgt[:, 1:]

            logits = model(src, dec_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            mask = dec_target.ne(PAD_IDX)
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

        train_loss = total_loss / max(total_tokens, 1)
        train_ppl = math.exp(min(train_loss, 20))

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_token_sum = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                dec_input = tgt[:, :-1]
                dec_target = tgt[:, 1:]
                logits = model(src, dec_input)
                loss = criterion(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))
                mask = dec_target.ne(PAD_IDX)
                val_loss_sum += loss.item() * mask.sum().item()
                val_token_sum += mask.sum().item()

        val_loss = val_loss_sum / max(val_token_sum, 1)
        val_ppl = math.exp(min(val_loss, 20))

        epoch_time = time.time() - epoch_start
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_ppl"].append(train_ppl)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["lr"].append(current_lr)

        print(f"  [{name}] Epoch {epoch:3d}/{epochs} ({epoch_time:.1f}s) | LR: {current_lr:.6f} "
              f"| Train Loss: {train_loss:.4f} PPL: {train_ppl:.2f} "
              f"| Val Loss: {val_loss:.4f} PPL: {val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if model_config is not None:
                best_path = os.path.join(checkpoint_dir, f"{name}_best.pt")
                _save_checkpoint(model, model_config, history, best_path)
            print(f"    -> New best val (loss: {best_val_loss:.4f}, PPL: {val_ppl:.2f})")
        else:
            patience_counter += 1
            print(f"    -> No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"    -> Early stopping at epoch {epoch}")
            break

    # Always save final weights (even if early stopped or crashed later)
    if model_config is not None:
        final_path = os.path.join(checkpoint_dir, f"{name}_latest.pt")
        _save_checkpoint(model, model_config, history, final_path)
        print(f"    -> Saved final weights to {final_path}")

    return history


def _test_eval(model: nn.Module, test_loader, device: str) -> dict:
    """Evaluate model on test set, return loss, PPL, and timing."""
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    total_loss = 0.0
    total_tokens = 0
    start = time.time()

    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            dec_input = tgt[:, :-1]
            dec_target = tgt[:, 1:]
            logits = model(src, dec_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))
            mask = dec_target.ne(PAD_IDX)
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

    elapsed = time.time() - start
    test_loss = total_loss / max(total_tokens, 1)
    test_ppl = math.exp(min(test_loss, 20))
    return {"test_loss": test_loss, "test_ppl": test_ppl, "test_time": elapsed}


def compare(
    epochs: int = 30,
    batch_size: int = 64,
    seq_len: int = 64,
    lr: float = 3e-4,
    patience: int = 4,
    warmup_epochs: int = 5,
    plot_dir: str = "plots",
    checkpoint_dir: str = "checkpoints",
) -> dict:
    """Benchmark CurioNet vs Transformer on WikiText-2.

    Both models are configured to have ~300K parameters.
    Runs on GPU (CUDA required).

    Returns dict with full results and saves comparison plots.
    """
    # Force GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. No GPU detected.")
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}\n")

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load WikiText-2
    print("Loading WikiText-2...")
    train_ds = WikiText2Dataset(split="train", seq_len=seq_len, max_chars=500000)
    val_ds = WikiText2Dataset(split="validation", seq_len=seq_len, max_chars=100000)
    test_ds = WikiText2Dataset(split="test", seq_len=seq_len, max_chars=100000)

    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")
    print(f"Vocab size:    {VOCAB_SIZE}")
    print()

    # --- CurioNet (~300K params) ---
    # dim=40, num_layers=2, num_heads=4 → ~293K params
    curionet_config = {
        "type": "curionet", "dim": 40, "num_layers": 2, "num_heads": 4,
        "src_vocab_size": VOCAB_SIZE, "tgt_vocab_size": VOCAB_SIZE,
        "dropout": 0.1, "padding_idx": PAD_IDX,
    }
    curionet = CurioNet(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        dim=40,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        padding_idx=PAD_IDX,
        max_len=seq_len + 4,
    ).to(device)
    curionet_params = _count_params(curionet)

    # --- Transformer (~300K params) ---
    # dim=64, num_layers=2, num_heads=4 → ~309K params
    transformer_config = {
        "type": "transformer", "dim": 64, "num_layers": 2, "num_heads": 4,
        "src_vocab_size": VOCAB_SIZE, "tgt_vocab_size": VOCAB_SIZE,
        "dropout": 0.1, "padding_idx": PAD_IDX,
    }
    transformer = TransformerSeq2Seq(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        padding_idx=PAD_IDX,
        max_len=seq_len + 4,
    ).to(device)
    transformer_params = _count_params(transformer)

    print("=" * 70)
    print(f"{'Model':<20} {'Parameters':>15}")
    print("-" * 70)
    print(f"{'CurioNet':<20} {curionet_params:>15,}")
    print(f"{'Transformer':<20} {transformer_params:>15,}")
    print("=" * 70)
    print()

    # --- Train or load CurioNet ---
    curionet_ckpt = os.path.join(checkpoint_dir, "curionet_best.pt")
    if os.path.exists(curionet_ckpt):
        print("=" * 70)
        print("Loading CurioNet from checkpoint (skipping training)")
        print("=" * 70)
        curionet_history = _load_checkpoint(curionet, curionet_ckpt, device)
    else:
        print("=" * 70)
        print("Training CurioNet (curiosity-based)")
        print("=" * 70)
        curionet_history = _train_one(
            curionet, "curionet", train_loader, val_loader,
            epochs, lr, device, patience, warmup_epochs,
            checkpoint_dir=checkpoint_dir, model_config=curionet_config,
        )

    # --- Train or load Transformer ---
    transformer_ckpt = os.path.join(checkpoint_dir, "transformer_best.pt")
    if os.path.exists(transformer_ckpt):
        print("\n" + "=" * 70)
        print("Loading Transformer from checkpoint (skipping training)")
        print("=" * 70)
        transformer_history = _load_checkpoint(transformer, transformer_ckpt, device)
    else:
        print("\n" + "=" * 70)
        print("Training Transformer (attention-based)")
        print("=" * 70)
        transformer_history = _train_one(
            transformer, "transformer", train_loader, val_loader,
            epochs, lr, device, patience, warmup_epochs,
            checkpoint_dir=checkpoint_dir, model_config=transformer_config,
        )

    # --- Test evaluation ---
    print("\n" + "=" * 70)
    print("Test Set Evaluation (WikiText-2 test split)")
    print("=" * 70)

    curionet_test = _test_eval(curionet, test_loader, device)
    transformer_test = _test_eval(transformer, test_loader, device)

    print(f"\n{'Metric':<25} {'CurioNet':>15} {'Transformer':>15}")
    print("-" * 60)
    print(f"{'Test Loss':<25} {curionet_test['test_loss']:>15.4f} {transformer_test['test_loss']:>15.4f}")
    print(f"{'Test PPL':<25} {curionet_test['test_ppl']:>15.2f} {transformer_test['test_ppl']:>15.2f}")
    print(f"{'Test Time (s)':<25} {curionet_test['test_time']:>15.2f} {transformer_test['test_time']:>15.2f}")
    print(f"{'Parameters':<25} {curionet_params:>15,} {transformer_params:>15,}")
    print()

    # Determine winner
    if curionet_test['test_ppl'] < transformer_test['test_ppl']:
        winner = "CurioNet"
        margin = transformer_test['test_ppl'] - curionet_test['test_ppl']
    else:
        winner = "Transformer"
        margin = curionet_test['test_ppl'] - transformer_test['test_ppl']

    print(f"Winner: {winner} (PPL margin: {margin:.2f})")
    print()

    # --- Comparison plots ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("CurioNet vs Transformer — WikiText-2 Benchmark", fontsize=16, fontweight="bold")

    # Train Loss
    axes[0, 0].plot(curionet_history["epoch"], curionet_history["train_loss"], "b-o", ms=2, label="CurioNet")
    axes[0, 0].plot(transformer_history["epoch"], transformer_history["train_loss"], "r-s", ms=2, label="Transformer")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Val Loss
    axes[0, 1].plot(curionet_history["epoch"], curionet_history["val_loss"], "b-o", ms=2, label="CurioNet")
    axes[0, 1].plot(transformer_history["epoch"], transformer_history["val_loss"], "r-s", ms=2, label="Transformer")
    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Val PPL
    axes[0, 2].plot(curionet_history["epoch"], curionet_history["val_ppl"], "b-o", ms=2, label="CurioNet")
    axes[0, 2].plot(transformer_history["epoch"], transformer_history["val_ppl"], "r-s", ms=2, label="Transformer")
    axes[0, 2].set_title("Validation PPL")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("PPL")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Test PPL bar chart
    axes[1, 0].bar(["CurioNet", "Transformer"],
                   [curionet_test["test_ppl"], transformer_test["test_ppl"]],
                   color=["#2471a3", "#cb4335"])
    axes[1, 0].set_title("Test PPL (lower = better)")
    axes[1, 0].set_ylabel("PPL")
    for i, v in enumerate([curionet_test["test_ppl"], transformer_test["test_ppl"]]):
        axes[1, 0].text(i, v + 0.5, f"{v:.2f}", ha="center", fontsize=12)

    # Param count
    axes[1, 1].bar(["CurioNet", "Transformer"],
                   [curionet_params, transformer_params],
                   color=["#2471a3", "#cb4335"])
    axes[1, 1].set_title("Parameter Count")
    axes[1, 1].set_ylabel("Params")
    for i, v in enumerate([curionet_params, transformer_params]):
        axes[1, 1].text(i, v + max(curionet_params, transformer_params) * 0.01, f"{v:,}", ha="center", fontsize=11)

    # Test time
    axes[1, 2].bar(["CurioNet", "Transformer"],
                   [curionet_test["test_time"], transformer_test["test_time"]],
                   color=["#2471a3", "#cb4335"])
    axes[1, 2].set_title("Test Inference Time (s)")
    axes[1, 2].set_ylabel("Seconds")
    for i, v in enumerate([curionet_test["test_time"], transformer_test["test_time"]]):
        axes[1, 2].text(i, v + 0.1, f"{v:.2f}s", ha="center", fontsize=12)

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "wikitext2_benchmark.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Benchmark plot saved to: {plot_path}")

    # --- Sample generation comparison ---
    print("\n--- Sample Generation ---")
    curionet.eval()
    transformer.eval()

    sample_text = "the first century of the roman empire"
    src = torch.tensor([encode(sample_text)], dtype=torch.long).to(device)

    curio_out = curionet.generate(src, max_len=40, bos_idx=BOS_IDX, eos_idx=EOS_IDX)
    trans_out = transformer.generate(src, max_len=40, bos_idx=BOS_IDX, eos_idx=EOS_IDX)

    print(f"  Input:       {sample_text}")
    print(f"  CurioNet:    {decode(curio_out[0].cpu().tolist())}")
    print(f"  Transformer: {decode(trans_out[0].cpu().tolist())}")

    return {
        "curionet": {
            "params": curionet_params,
            "history": curionet_history,
            "test": curionet_test,
        },
        "transformer": {
            "params": transformer_params,
            "history": transformer_history,
            "test": transformer_test,
        },
        "winner": winner,
        "ppl_margin": margin,
    }


if __name__ == "__main__":
    compare()
