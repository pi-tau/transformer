import numpy as np
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer as described in section 3.2 of the paper
    "Attention is all you need"
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self, in_dims, out_dims, n_heads, attn_dropout=0.):
        """Init a multi-head attention layer.

        Args:
            in_dims: int
                Number of features in the input.
            out_dims: int
                Number of features in the output encodings.
            n_heads: int
                Number of attention heads.
            attn_dropout: float, optional
                Dropout value for the attention scores. Default: 0.
        """
        super().__init__()
        assert out_dims % n_heads == 0, "out dims must be divisible by num heads"
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.n_heads = n_heads

        self.qkv = nn.Linear(in_dims, 3 * out_dims, bias=False)
        self.Wo = nn.Linear(out_dims, out_dims)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # Initialize the Q, K, and V matrices using Xavier initialization to
        # make sure that the produced queries and keys have unit std. Note that
        # we are manually setting the initialization parameters, because we are
        # using a single batched layer containing the three matrices.
        nn.init.normal_(self.qkv.weight, mean=0., std=np.sqrt(2 / (in_dims+out_dims)))

        # Initialize the output layer to have Xavier weights and zero biases.
        nn.init.xavier_normal_(self.Wo.weight)
        nn.init.zeros_(self.Wo.bias)

    def forward(self, x, mask=None):
        """Transform the input using self-attention.
        This layer uses scaled dot-product attention. Each input vector is
        transformed into three separate representations: query, key and value.
        All three have the same dimension: `out_dims // n_head`.

        Args:
            x: torch.Tensor
                Tensor of shape (B, T, D).
            mask: torch.Tensor, optional
                Boolean tensor of shape (B, T) used for masking specific entries
                from the input tensor. Default: None.

        Returns:
            out: torch.Tensor
                Tensor of shape (B, T, embed_dim), giving the encodings from the
                self-attention layer.
            attn: torch.Tensor
                Tensor of shape (B, n_head, T, T) giving the pairwise
                self-attention probability scores from each head.
        """
        B, T, _ = x.shape

        # Compute the queries, keys and values using a single forward pass.
        queries, keys, values = self.qkv(x).chunk(chunks=3, dim=2)

        # Each is reshaped to (B, T, nh, hid) and is transposed to (B, nh, T, hid).
        queries = queries.view(B, T, self.n_heads, -1).transpose(1, 2)  # X @ Q
        keys = keys.view(B, T, self.n_heads, -1).transpose(1, 2)        # X @ K
        values = values.view(B, T, self.n_heads, -1).transpose(1, 2)    # X @ V

        # Compute the attentions scores by multiplying Qs and Ks.
        # Both queries and keys are 4D tensors and `matmul` will perform a
        # batched matrix multiplication over the last two dimensions.
        attn = torch.matmul(queries, keys.transpose(2, 3)) # XV @ (XK)^T and shape (B, nh, T, T)

        # Scale the attentions scores with the sqrt of the dimension of the keys.
        # We want to scale the att scores, because dot products grow rapidly and
        # the softmax will saturate to 1 for one of the elements and 0 for all
        # others.
        # A more rigorous explanation is that we want to maintain the variance of
        # the attentions scores to be the same as the variance of the Qs and Ks.
        # Assuming: Q ~ N(0, s^2) and K ~ N(0, s^2), then Var(Q^T @ K) = s^4 * dk.
        # If we do not scale the variance back to s^2 some score might attain a
        # very large value and the softmax will saturate to 1. Since, originally
        # we have s close to 1, we need to scale by sqrt(dk).
        dk = keys.shape[-1]
        attn /=  np.sqrt(dk)
        if mask is not None:
            # Set the attention logits for masked inputs to a very low value.
            # Unsqueeze the mask tensor to match the shape of the attention logits.
            attn = attn.masked_fill_(mask[:, None, :, None], -1e-10)
        attn = torch.softmax(attn, dim=-1) # shape (B, nh, T, T)

        # It looks strange that we are applying dropout directly to the attention.
        # This means that our attention vector will most probably not sum to 1.
        # The paper never mentions or explains this but it is used in the official
        # implementation, including BERT and GPT.
        # However, note that during evaluation dropout is not applied so we are
        # probably fine.
        attn = self.attn_dropout(attn)

        # Compute the outputs from the multiple attention heads and concatenate
        # them along the hidden dimensions. Then forward through the output layer.
        z = torch.matmul(attn, queries)      # shape (B, nh, T, hid)
        z = z.transpose(1, 2).reshape(B, T, -1)
        out = self.Wo(z)                    # shape (B, T, out_dims)

        return out, attn

#