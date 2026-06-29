import torch
import torch.nn as nn
import math


class WonderGenerator(nn.Module):
    """Generates wonder states — the model's internal curiosity.

    This is NOT attention: it doesn't weight or select from existing inputs.
    This is NOT a detector: it doesn't classify or identify patterns.
    This is NOT a score: it doesn't assign numerical values to inputs.

    It GENERATES entirely new internal states — like a human
    who reads something and then wonders 'but what about...?'
    """

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.expand = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.contract = nn.Linear(dim * expansion, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.expand(x)
        h = self.act(h)
        h = self.contract(h)
        g = torch.sigmoid(self.gate(x))
        wonder = h * g
        return self.norm(wonder)


class WonderConv(nn.Module):
    """Convolutional processing block for wonder states.

    The 'thinking' phase — where the model processes its own curiosity.
    """

    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x + residual
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)


class InsightExtractor(nn.Module):
    """Extracts insights from processed wonder states.

    Insights are the 'aha!' moments — the model synthesizes
    what it learned from its curiosity into actionable knowledge.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.norm(x)


class CuriosityLayer(nn.Module):
    """A single curiosity layer — the primary building block of CurioNet.

    Replaces what an attention layer does in a Transformer:
    - Transformer: Q·Kᵀ → softmax → weighted sum of V (re-weighting existing info)
    - CurioNet: generate wonder → process wonder → extract insights → integrate

    The curiosity layer GENERATES new internal states rather than
    re-weighting existing ones. This is the fundamental difference.

    Pipeline:
    1. Generate wonder from current representation
    2. Process wonder through 'thinking' conv layers
    3. Extract insights from processed wonder
    4. Integrate insights back via gated residual

    Args:
        dim: Hidden dimension.
        num_wonder_convs: Conv layers for processing wonder.
        kernel_size: Kernel size for wonder convs.
        dropout: Dropout rate.
        curiosity_budget: Wonder cycles per layer (higher = more curious).
    """

    def __init__(
        self,
        dim: int,
        num_wonder_convs: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        curiosity_budget: int = 1,
    ):
        super().__init__()
        self.curiosity_budget = curiosity_budget
        self.wonder_gen = WonderGenerator(dim)
        self.wonder_convs = nn.ModuleList([
            WonderConv(dim, kernel_size, dropout) for _ in range(num_wonder_convs)
        ])
        self.insight = InsightExtractor(dim, dropout)
        self.integrate_gate = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> tuple:
        # x: (batch, seq_len, dim)
        residual = x
        current = x
        last_wonder = None

        for _ in range(self.curiosity_budget):
            # 1. Generate wonder
            wonder = self.wonder_gen(current)
            last_wonder = wonder

            # 2. Process wonder through 'thinking' conv layers
            w = wonder.transpose(1, 2)  # (batch, dim, seq_len)
            for layer in self.wonder_convs:
                w = layer(w)

            # 3. Extract insights
            insights = self.insight(w.transpose(1, 2))

            # 4. Integrate insights via gated residual
            current = current + insights

        # Gated integration with residual
        integrated = self.integrate_gate(torch.cat([current, residual], dim=-1))
        output = self.norm(integrated + residual)

        return output, last_wonder


class CuriosityBlock(nn.Module):
    """Curiosity layer + feed-forward — like a Transformer block but curiosity-based.

    Transformer block:  [Attention → FFN]
    CurioNet block:     [Curiosity → FFN]

    The FFN is the same (Linear → GELU → Linear), but the core
    mechanism is curiosity instead of attention.

    Args:
        dim: Hidden dimension.
        ff_dim: Feed-forward expansion dimension.
        num_wonder_convs: Conv layers in curiosity.
        kernel_size: Conv kernel size.
        dropout: Dropout rate.
        curiosity_budget: Wonder cycles.
    """

    def __init__(
        self,
        dim: int,
        ff_dim: int = None,
        num_wonder_convs: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        curiosity_budget: int = 1,
    ):
        super().__init__()
        if ff_dim is None:
            ff_dim = dim * 4
        self.curiosity = CuriosityLayer(
            dim, num_wonder_convs, kernel_size, dropout, curiosity_budget
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> tuple:
        x, wonder = self.curiosity(x)
        x = self.ffn_norm(x + self.ffn(x))
        return x, wonder
