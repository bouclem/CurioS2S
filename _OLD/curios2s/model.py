import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .curiosity import CuriosityDrive


class PositionalEmbedding(nn.Module):
    """Learned positional embeddings as in Gehring et al. (2017)."""

    def __init__(self, vocab_size: int, dim: int, max_len: int = 512, padding_idx: int = 0):
        super().__init__()
        self.padding_idx = padding_idx
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(max_len, dim, padding_idx=padding_idx)
        self.scale = math.sqrt(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        batch, seq_len = x.size()
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)
        mask = x.ne(self.padding_idx).long()
        pos = pos * mask
        emb = self.token_emb(x) * self.scale + self.pos_emb(pos)
        return emb * mask.unsqueeze(-1).float()


class GLU(nn.Module):
    """Gated Linear Unit: (A, B) -> A * sigmoid(B).

    A single conv doubles the channel count, then we split in half.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class ConvBlock(nn.Module):
    """One convolutional block: Conv1d -> GLU -> residual.

    For the encoder, padding is symmetric (both sides).
    For the decoder, padding is left-only (causal).
    """

    def __init__(self, dim: int, kernel_size: int = 3, causal: bool = False, dropout: float = 0.1):
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, 2 * dim, kernel_size=kernel_size, padding=0)
        self.glu = GLU(dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim, seq_len)
        residual = x
        if self.causal:
            # Left-pad for causal convolution
            pad = (self.kernel_size - 1, 0)
        else:
            pad = ((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2)
        x = F.pad(x, pad)
        x = self.conv(x)
        x = self.glu(x)
        x = self.dropout(x)
        # Residual + scaling (as in the paper)
        return (x + residual) * self.scale


class ConvEncoder(nn.Module):
    """Stack of convolutional blocks with positional embeddings."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        max_len: int = 512,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.embedding = PositionalEmbedding(vocab_size, dim, max_len, padding_idx)
        self.input_proj = nn.Linear(dim, dim)
        self.layers = nn.ModuleList(
            [ConvBlock(dim, kernel_size, causal=False, dropout=dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(dim, dim)
        self.curiosity = CuriosityDrive(dim, num_wonder_layers=2, kernel_size=kernel_size, dropout=dropout)

    def forward(self, src: torch.Tensor) -> tuple:
        # src: (batch, src_len)
        mask = src.ne(self.embedding.padding_idx)  # (batch, src_len)
        x = self.embedding(src)  # (batch, src_len, dim)
        x = self.input_proj(x)

        # Store embeddings for attention (used by decoder)
        emb = x * mask.unsqueeze(-1).float()

        # Transpose to (batch, dim, seq_len) for conv
        x = x.transpose(1, 2)
        conv_outputs = []
        for layer in self.layers:
            x = layer(x)
            conv_outputs.append(x.transpose(1, 2))  # (batch, seq_len, dim)

        # Curiosity: generate wonder, process it, enrich representation
        x, wonder = self.curiosity(x)

        # Last conv output, projected
        x = x.transpose(1, 2)
        x = self.output_proj(x) * mask.unsqueeze(-1).float()

        return x, emb, mask, conv_outputs


class ConvAttention(nn.Module):
    """Attention from decoder conv output to encoder states.

    Combines encoder embeddings and encoder conv outputs
    as described in Gehring et al. (2017).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dec_proj = nn.Linear(dim, dim)
        self.enc_proj = nn.Linear(dim * 2, dim)
        self.combine = nn.Linear(dim * 3, dim)

    def forward(
        self,
        dec_state: torch.Tensor,
        enc_conv_out: torch.Tensor,
        enc_emb: torch.Tensor,
        enc_mask: torch.Tensor,
    ) -> tuple:
        # dec_state: (batch, tgt_len, dim)
        # enc_conv_out: (batch, src_len, dim)  -- from encoder's last conv layer
        # enc_emb: (batch, src_len, dim)       -- encoder embeddings
        # enc_mask: (batch, src_len)

        batch, tgt_len, dim = dec_state.size()
        src_len = enc_conv_out.size(1)

        # Project decoder state
        q = self.dec_proj(dec_state)  # (batch, tgt_len, dim)

        # Combine encoder conv output + embedding, then project
        enc_combined = torch.cat([enc_conv_out, enc_emb], dim=-1)  # (batch, src_len, 2*dim)
        k = self.enc_proj(enc_combined)  # (batch, src_len, dim)

        # Scaled dot-product attention
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(dim)  # (batch, tgt_len, src_len)

        # Mask padding positions
        scores = scores.masked_fill(~enc_mask.unsqueeze(1), -1e9)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, tgt_len, src_len)

        # Context vector = weighted sum of (encoder conv output + embedding)
        context = torch.bmm(attn_weights, enc_combined)  # (batch, tgt_len, 2*dim)

        # Combine context with decoder state
        output = self.combine(torch.cat([context, dec_state], dim=-1))  # (batch, tgt_len, dim)
        return output, attn_weights


class ConvDecoder(nn.Module):
    """Stack of convolutional blocks with attention to encoder."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        max_len: int = 512,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.padding_idx = padding_idx
        self.embedding = PositionalEmbedding(vocab_size, dim, max_len, padding_idx)
        self.input_proj = nn.Linear(dim, dim)
        self.layers = nn.ModuleList(
            [ConvBlock(dim, kernel_size, causal=True, dropout=dropout) for _ in range(num_layers)]
        )
        self.attentions = nn.ModuleList([ConvAttention(dim) for _ in range(num_layers)])
        self.curiosity = CuriosityDrive(dim, num_wonder_layers=2, kernel_size=kernel_size, dropout=dropout)
        self.output_proj = nn.Linear(dim, vocab_size)

    def forward(self, tgt: torch.Tensor, enc_out: tuple, teacher_forcing: bool = True) -> torch.Tensor:
        # tgt: (batch, tgt_len)
        # enc_out: (enc_conv_out, enc_emb, enc_mask, conv_outputs) from encoder
        enc_conv_out, enc_emb, enc_mask, enc_conv_outputs = enc_out

        tgt_mask = tgt.ne(self.padding_idx)
        x = self.embedding(tgt)
        x = self.input_proj(x)
        x = x.transpose(1, 2)  # (batch, dim, tgt_len)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Convert to (batch, tgt_len, dim) for attention
            x_att = x.transpose(1, 2)
            attended, _ = self.attentions[i](
                x_att, enc_conv_out, enc_emb, enc_mask
            )
            # Add attention output as residual
            x = (x + attended.transpose(1, 2)) * math.sqrt(0.5)

        # Curiosity: generate wonder, process it, enrich representation
        x, wonder = self.curiosity(x)

        x = x.transpose(1, 2)  # (batch, tgt_len, dim)
        logits = self.output_proj(x)
        return logits


class CurioS2S(nn.Module):
    """CurioS2S: Convolutional Seq2Seq with Curiosity Drive.

    Based on ConvS2S (Gehring et al., 2017) with an added CuriosityDrive
    that generates wonder states — the model's internal curiosity that
    makes it think deeper, want answers, explore further.

    Curiosity is NOT attention, NOT a detector, NOT a score.
    It is a generative drive that creates new internal states.

    Args:
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        dim: Hidden dimension (default 256).
        num_layers: Number of conv layers in encoder/decoder (default 4).
        kernel_size: Convolution kernel width (default 3).
        dropout: Dropout rate (default 0.1).
        max_len: Maximum sequence length (default 512).
        padding_idx: Padding token index (default 0).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim: int = 256,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        max_len: int = 512,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.encoder = ConvEncoder(
            src_vocab_size, dim, num_layers, kernel_size, dropout, max_len, padding_idx
        )
        self.decoder = ConvDecoder(
            tgt_vocab_size, dim, num_layers, kernel_size, dropout, max_len, padding_idx
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(src)
        logits = self.decoder(tgt, enc_out)
        return logits

    @torch.no_grad()
    def generate(self, src: torch.Tensor, max_len: int = 50, bos_idx: int = 1, eos_idx: int = 2) -> torch.Tensor:
        """Greedy decoding for inference."""
        self.eval()
        batch = src.size(0)
        device = src.device
        enc_out = self.encoder(src)

        tgt = torch.full((batch, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.decoder(tgt, enc_out)
            next_token = logits[:, -1, :].argmax(dim=-1)  # (batch,)
            next_token = next_token.masked_fill(finished, self.decoder.padding_idx)
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_idx)
            if finished.all():
                break

        return tgt

# Backward-compatible alias
ConvS2S = CurioS2S
