import torch
import torch.nn as nn

from multihead_attn import MultiHeadAttention


class DecoderLayer(nn.Module):
    """Transformer Decoder layer as described in section 3.1 of the paper
    "Attention is all you need"
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model, n_heads, dim_mlp=2048, dropout=0.0):
        """Init an Decoder layer for the Transformer model.

        Args:
            d_model: int
                Size of the decoder layer. Because the model uses residual
                connections, both the input and the output will have the same size.
            n_heads: int
                Number of heads for multi-head attention.
            dim_mlp: int, optional
                Dimension of the hidden layer of the MLP network. Default: 2048.
            dropout: int, optional
                Dropout rate applied for the multi-head attention, as well as
                to the outputs of the sub-layers. Default: 0.
        """
        super().__init__()
        assert d_model % n_heads == 0, "model dims must be divisible by num heads"

        # The decoder layer has three sub-layers.
        # Residual connections are applied around each of the three sub-layers,
        # followed by layer normalization. In addition the output of each of the
        # sub-layers is forwarded through a dropout layer to increase
        # regularization. In addition the output of each of the sub-layers is
        # forwarded through a dropout layer to increase regularization.

        # The first sub-layer is a casually masked multi-headed self-attention.
        self.self_attn = MultiHeadAttention(
            in_dim=d_model,
            qk_dim=d_model,
            v_dim=d_model,
            n_heads=n_heads,
            attn_dropout=dropout,
        )
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)

        # The second is a cross-attention layer, attending the output of the
        # encoder stack.
        self.cross_attn = MultiHeadAttention(
            in_dim=d_model,
            qk_dim=d_model,
            v_dim=d_model,
            n_heads=n_heads,
            attn_dropout=dropout,
        )
        self.cross_attn_dropout = nn.Dropout(dropout)
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # And the third is a position-wise fully-connected network.
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, d_model),
        )
        self.mlp_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(d_model)

    def forward(self, x, mem):
        """Decode the input using the Transformer Decoder layer.

        Args:
            x: torch.Tensor
                Tensor of shape (B, T, D).
            mem: torch.Tensor
                Tensor of shape (B, T_enc, D), giving the encoder outputs.

        Returns:
            r: torch.Tensor
                Tensor of shape (B, T, D), giving the encodings of the input.
        """
        # Apply self-attention, then add the residual connection and normalize.
        z, _ = self.self_attn(x, x, x, is_causal=True)
        z = self.self_attn_norm(x + self.self_attn_dropout(z))

        # Apply cross-attention, then add the residual connection and normalize.
        # Use the decoded sequence as queries and the encoder outputs as keys
        # and values (like memory).
        c, _ = self.cross_attn(z, mem, mem)
        c = self.cross_attn_norm(z + self.cross_attn_dropout(c))

        # Run through the position-wise network, then add the residual and normalize.
        r = self.mlp(c)
        r = self.mlp_norm(c + self.mlp_dropout(r))

        return r
