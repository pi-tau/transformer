import torch
import torch.nn as nn

from multihead_attn import MultiHeadAttention


class DecoderBlock(nn.Module):
    """Transformer Decoder block as described in section 3.1 of the paper
    "Attention is all you need"
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model, n_heads, dim_mlp, dropout):
        """Init an Decoder block for the Transformer model.

        Args:
            d_model: int
                Size of the decoder block. Because the model uses residual
                connections, both the input and the output will have the same size.
            n_heads: int
                Number of heads for multi-head attention.
            dim_mlp: int
                Dimension of the hidden layer of the MLP network.
            dropout: float
                Dropout rate applied for the multi-head attention, as well as
                to the outputs of the sub-layers.
        """
        super().__init__()
        assert d_model % n_heads == 0, "model dims must be divisible by num heads"

        # The decoder block has three sub-layers.
        # Residual connections are applied around each of the three sub-layers.
        # Before applying the sub-layer we will normalize the input, as proposed
        # in Ruibin Xiong et al. (http://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf).
        # This supports better gradient flow and removes the need for a warm-up
        # stage. In addition the output of each of the sub-layers is forwarded
        # through a dropout layer to increase regularization.

        # The first sub-layer is a casually masked multi-headed self-attention.
        self.self_attn = MultiHeadAttention(
            in_dim=d_model,
            qk_dim=d_model,
            v_dim=d_model,
            out_dim=d_model,
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
            out_dim=d_model,
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

    def forward(self, x, mem, mem_mask=None):
        """Decode the input using the Transformer Decoder block.

        Args:
            x: torch.Tensor
                Tensor of shape (B, T, D).
            mem: torch.Tensor
                Tensor of shape (B, T_enc, D), giving the encoder outputs.
            mem_mask: torch.Tensor
                Boolean tensor of shape (B, T_enc), indicating which elements of
                the encoder outputs should be masked. A value of True indicates
                that the element *should* take part in the computation.
                Default: None.

        Returns:
            r: torch.Tensor
                Tensor of shape (B, T, D), giving the encodings of the input.
        """
        # Normalize, apply causal self-attention, then add residual connection.
        # For causal self-attention we use a causal mask.
        _, T, _ = x.shape
        causal_mask = torch.ones(1, T, T, dtype=torch.bool).tril().to(x.device)
        x = self.self_attn_norm(x)
        z, _ = self.self_attn(x, x, x, mask=causal_mask)
        z = x + self.self_attn_dropout(z)

        # Normalize, apply cross-attention, then add the residual connection.
        # Use the decoded sequence as queries and the encoder outputs as keys
        # and values (like memory). Broadcasting the mem mask along the second
        # to last dim is enough for cross-attn.
        if mem_mask is not None: mem_mask = mem_mask.unsqueeze(dim=-2)
        z = self.cross_attn_norm(z)
        c, _ = self.cross_attn(z, mem, mem, mask=mem_mask)
        c = z + self.cross_attn_dropout(c)

        # Normalize, run through the position-wise network, then add the residual.
        c = self.mlp_norm(c)
        r = self.mlp(c)
        r = c + self.mlp_dropout(r)

        return r

#