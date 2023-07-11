import torch.nn as nn

from multihead_attn import MultiHeadAttention


class EncoderLayer(nn.Module):
    """Transformer Encoder layer as described in section 3.1 of the paper
    "Attention is all you need"
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model, n_heads, dim_mlp=2048, dropout=0.0):
        """Init an Encoder layer for the Transformer model.

        Args:
            d_model: int
                Size of the encoder layer. Because the model uses residual
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

        # The encoder layer has two sub-layers.
        # Residual connections are applied around each of the two sub-layers.
        # Before applying the sub-layer we will normalize the input, as proposed
        # in Ruibin Xiong et al. (http://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf).
        # This supports better gradient flow and removes the need for a warm-up
        # stage. In addition the output of each of the sub-layers is forwarded
        # through a dropout layer to increase regularization.

        # The first sub-layer is a multi-headed self-attention.
        self.attn = MultiHeadAttention(
            in_dim=d_model,
            qk_dim=d_model,
            v_dim=d_model,
            n_heads=n_heads,
            attn_dropout=dropout,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_model)

        # The second sub-layer is a position-wise fully-connected network.
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, d_model),
        )
        self.mlp_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """Encode the input using the Transformer Encoder layer.

        Args:
            x: torch.Tensor
                Tensor of shape (B, T, D).
            mask: torch.Tensor
                Boolean tensor of shape (B, T), indicating which elements of
                the input should be masked. A value of True indicates that the
                element *should* take part in the computation. Default: None.

        Returns:
            r: torch.Tensor
                Tensor of shape (B, T, D), giving the encodings of the input.
        """
        # Normalize, apply self-attention, then add the residual connection.
        # Broadcasting the mask along the last dim is enough for self-attention.
        if mask is not None: mask = mask.unsqueeze(dim=-1)
        x = self.attn_norm(x)
        z, _ = self.attn(x, x, x, mask=mask)
        z = x + self.attn_dropout(z)

        # Normalize, run through the position-wise network, then add the residual.
        z = self.mlp_norm(z)
        r = self.mlp(z)
        r = z + self.mlp_dropout(r)

        return r

#