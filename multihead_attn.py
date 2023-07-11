import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer as described in section 3.2 of the paper
    "Attention is all you need"
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self, in_dim, qk_dim, v_dim, n_heads, attn_dropout=0.):
        """Init a multi-head attention layer.

        Args:
            in_dim: int
                Number of features in the input.
            qk_dim: int
                Number of features in the query and key embeddings.
            v_dim: int
                Number of features in the value embedding. The number of output
                features will match the number of features in the values.
            n_heads: int
                Number of attention heads.
            attn_dropout: float, optional
                Dropout value for the attention scores. Default: 0.
        """
        super().__init__()
        assert qk_dim % n_heads == 0, "query and key dims must be divisible by num heads"
        assert v_dim % n_heads == 0, "value dim must be divisible by num heads"
        self.n_heads = n_heads
        self.dropout_p = attn_dropout

        self.Q = nn.Linear(in_dim, qk_dim, bias=False)
        self.K = nn.Linear(in_dim, qk_dim, bias=False)
        self.V = nn.Linear(in_dim, v_dim, bias=False)
        self.Wo = nn.Linear(v_dim, v_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # Initialize the Q and K matrices using Xavier initialization to make
        # sure that the produced queries and keys have unit std.
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)

        # The V and Wo matrices will use the default initialization for the
        # weights. The Wo biases will be set to zero.
        nn.init.zeros_(self.Wo.bias)

        # Create a mask buffer for causal attention. We assume that the maximum
        # sequence length will be 1024.
        M = 1024
        self.register_buffer(
            "mask",
            torch.ones(M, M, dtype=torch.bool).tril().view(1, 1, M, M),
        )

    def forward(self, queries, keys, values, is_causal=False):
        """Compute the dot-product attention scores between the queries and keys
        and combine the values using these scores. For computing self-attention
        use as: `forward(x, x, x)`.

        Args:
            queries: torch.Tensor
                Tensor of shape (B, T, D) holding the queries.
            keys: torch.Tensor
                Tensor of shape (B, T, D) holding the keys.
            values: torch.Tensor
                Tensor of shape (B, T, D) holding the values.
            is_causal: bool, optional
                If True, assumes causal attention masking. Default: False.

        Returns:
            out: torch.Tensor
                Tensor of shape (B, T, embed_dim), giving the encodings from the
                attention layer.
            attn: torch.Tensor
                Tensor of shape (B, n_head, T, T) giving the pairwise attention
                probability scores from each head.
        """
        B, T, _ = queries.shape

        # Compute the query, key and value embeddings.
        # For multi-head attention each embedding is reshaped into
        # (B, T, nh, hid) and is transposed to (B, nh, T, hid).
        q = self.Q(queries).view(B, T, self.n_heads, -1).transpose(1, 2) # X @ Q
        k = self.K(keys).view(B, T, self.n_heads, -1).transpose(1, 2)    # X @ K
        v = self.V(values).view(B, T, self.n_heads, -1).transpose(1, 2)  # X @ V

        # Compute the attentions scores by multiplying qs and ks.
        # Both are 4D tensors and `matmul` will perform a batched matrix
        # multiplication over the last two dimensions.
        attn = torch.matmul(q, k.transpose(2, 3)) # XQ @ (XK)^T and shape (B, nh, T, T)

        # Scale the attentions scores with the sqrt of the dimension of the keys.
        # We want to scale the attn scores, because dot products grow rapidly
        # and the softmax will saturate to 1 for one of the elements and 0 for
        # all others.
        # A more rigorous explanation is that we want to maintain the variance of
        # the attentions scores to be the same as the variance of the qs and ks.
        # Assuming: q ~ N(0, s^2) and k ~ N(0, s^2), then Var(q^T @ k) = s^4 * dk.
        # If we do not scale the variance back to s^2 some score might attain a
        # very large value and the softmax will saturate to 1. Since, originally
        # we have s close to 1, we need to scale by sqrt(dk).
        dk = k.shape[-1]
        attn /=  np.sqrt(dk)
        if is_causal:
            # Set the attention logits for masked inputs to a very low value.
            # Unsqueeze the mask tensor to match the shape of the attention logits.
            attn = attn.masked_fill_(self.mask[:, :, :T, :T], -1e-8)
        attn = torch.softmax(attn, dim=-1) # shape (B, nh, T, T)

        # It looks strange that we are applying dropout directly to the attention.
        # This means that our attention vector will most probably not sum to 1.
        # The paper never mentions or explains this but it is used in the official
        # implementation, including BERT and GPT.
        # However, note that during evaluation dropout is not applied so we are
        # probably fine.
        attn = self.attn_dropout(attn)

        # Combine the values using the attention probabilities. Stack the output
        # from the heads and forward through the output layer.
        z = torch.matmul(attn, v) # shape (B, nh, T, hid)
        z = z.transpose(1, 2).reshape(B, T, -1)
        out = self.Wo(z)          # shape (B, T, out_dims)

        return out, attn

#