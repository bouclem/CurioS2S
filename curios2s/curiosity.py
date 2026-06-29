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
    These wonder states represent questions the model wants
    answers to, directions it wants to explore further.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.contract = nn.Linear(dim * 4, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.expand(x)
        h = self.act(h)
        h = self.contract(h)
        g = torch.sigmoid(self.gate(x))
        return h * g


class WonderConv(nn.Module):
    """Convolutional processing block for wonder states.

    This is the 'thinking' phase — where the model processes
    its own curiosity. Separate from the main conv blocks.
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


class CuriosityDrive(nn.Module):
    """Curiosity mechanism for CurioS2S.

    Curiosity is a generative drive. It is NOT:
    - Attention (doesn't weight existing inputs)
    - A detector (doesn't classify or detect patterns)
    - A score (doesn't assign numerical values)

    What curiosity DOES:
    1. Generates wonder — new internal states representing the model's
       desire to explore, its questions, its 'what if?' thoughts
    2. Processes wonder through dedicated 'thinking' conv layers —
       the model ponders its own curiosity, thinks deeper
    3. Produces insights from wonder processing
    4. Integrates insights back, enriching the representation

    Like a human with lots of curiosity who:
    - Wants responses to questions
    - Tries to look more, think deeper
    - Generates new questions from what they know
    - Won't settle for surface-level understanding

    Args:
        dim: Hidden dimension.
        num_wonder_layers: Conv layers for processing wonder.
        kernel_size: Kernel size for wonder convs.
        dropout: Dropout rate.
        curiosity_budget: Number of wonder cycles (higher = more curious).
    """

    def __init__(
        self,
        dim: int,
        num_wonder_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        curiosity_budget: int = 1,
    ):
        super().__init__()
        self.curiosity_budget = curiosity_budget
        self.wonder_gen = WonderGenerator(dim)
        self.wonder_norm = nn.LayerNorm(dim)
        self.wonder_convs = nn.ModuleList([
            WonderConv(dim, kernel_size, dropout) for _ in range(num_wonder_layers)
        ])
        self.insight_proj = nn.Linear(dim, dim)
        self.insight_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> tuple:
        # x: (batch, dim, seq_len) — conv format
        # Returns: enriched x (batch, dim, seq_len), last_wonder (batch, seq_len, dim)

        x_seq = x.transpose(1, 2)  # (batch, seq_len, dim)
        current = x_seq
        last_wonder = None

        for _ in range(self.curiosity_budget):
            # 1. Generate wonder: "what am I curious about?"
            wonder = self.wonder_gen(current)
            wonder = self.wonder_norm(wonder)
            last_wonder = wonder

            # 2. Process wonder through 'thinking' conv layers
            w = wonder.transpose(1, 2)  # (batch, dim, seq_len)
            for layer in self.wonder_convs:
                w = layer(w)

            # 3. Generate insights from wonder processing
            insights = self.insight_proj(w.transpose(1, 2))
            insights = self.insight_norm(insights)

            # 4. Integrate insights: enriched representation
            current = current + insights

        return current.transpose(1, 2), last_wonder
