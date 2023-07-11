import torch.nn as nn

from multihead_attn import MultiHeadAttention


class EncoderLayer(nn.Module):
    """Transformer Encoder layer as described in section 3.1 of the paper
    "Attention is all you need"
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model, n_head, dim_mlp=2048, dropout=0.0):
        """Init an Encoder layer for the Transformer model.

        Args:
            d_model: int
                Size of the encoder layer. Because the model uses residual
                connections, both the input and the output will have the same size.
            n_head: int
                Number of heads for multi-head attention.
            dim_mlp: int, optional
                Dimension of the hidden layer of the MLP network. Default: 2048.
            dropout: int, optional
                Dropout rate applied for the multi-head attention, as well as
                to the outputs of the sub-layers. Default: 0.
        """
        super().__init__()
        assert d_model % n_head == 0, "model dims must be divisible by num heads"

        # The encoder layer has two sub-layers. The first is a multi-head attention,
        # and the second is a position-wise fully connected network. Residual
        # connections are applied around each of the two sub-layers, followed by
        # layer normalization.
        # In addition the output of each of the sub-layers is forwarded through
        # a dropout layer to increase regularization.
        self.attn = MultiHeadAttention(d_model, d_model, n_head, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_model)

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
            mask: torch.Tensor, optional
                Boolean tensor of shape (B, T) used for masking specific entries
                from the input tensor. Default: None.

        Returns:
            r: torch.Tensor
                Tensor of shape (B, T, D), giving the encodings of the input.
        """
        # Apply self-attention, then add the residual connection and normalize.
        z, _ = self.attn(x, mask)
        z = self.attn_norm(x + self.attn_dropout(z))

        # Run through the position-wise network, then add the residual and normalize.
        r = self.mlp(z)
        r = self.mlp_norm(z + self.mlp_dropout(r))

        return r

#