import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    """Token embedding layer with positional encoding."""

    def __init__(self, word_embed_weight, pos_embed_weight, scale, dropout):
        """Init an embedding layer.

        Args:
            word_embed_weights: torch.Tensor
                Word embedding matrix.
            pos_embed_weights: torch.Tensor
                Positional embedding matrix.
            scale: float
                Scaling factor for scaling the word embeddings before summing
                with the positional embeddings.
            dropout: float
                Dropout rate applied to the final embeddings.
        """
        super().__init__()
        max_len, _ = pos_embed_weight.shape
        self.max_len = max_len
        self.word_embed_weight = word_embed_weight
        self.pos_embed_weight = pos_embed_weight
        self.dropout = nn.Dropout(dropout)

        # Register a buffer with sequence positions. At runtime simply slice
        # the buffer to the requested size.
        self.register_buffer("positions", torch.arange(max_len).unsqueeze(dim=0))
        self.register_buffer("scale", torch.sqrt(torch.FloatTensor([scale])))

    def forward(self, x):
        _, T = x.shape
        if T > self.max_len: # explicitly check out-of-bound slicing
            raise RuntimeError("Sequence length exceeds the maximum allowed limit")

        pos = self.positions[:, :T]
        word_embed = F.embedding(x, self.word_embed_weight)
        pos_embed = F.embedding(pos, self.pos_embed_weight)
        embed = pos_embed + word_embed * self.scale
        return self.dropout(embed)

#