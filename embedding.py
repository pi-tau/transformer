import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """Embedding layer with positional encoding."""

    def __init__(self, vocab_size, embed_size, dropout=0., max_len=256):
        """Init an embedding layer.

        Args:
            vocab_size: int
                Size of the token vocabulary.
            embed_size: int
                Size of the embedding space.
            dropout: float, optional
                Dropout rate applied for embeddings. Default: 0.
            max_len: int, optional
                Maximum length of the token sequence. Default: 256.
        """
        super().__init__()
        # Use nn.Embedding. Similar to nn.Linear but accepts indices instead of
        # one-hot vectors.
        self.word_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(max_len, embed_size)
        self.dropout = nn.Dropout(dropout)

        # Register a buffer with sequence positions. At runtime simply slice
        # the buffer to the requested size.
        self.register_buffer("positions", torch.arange(max_len).unsqueeze(dim=0))

        # TODO: Explain why we need to scale the word embeddings.
        self.register_buffer("scale", torch.sqrt(torch.FloatTensor([embed_size])))

    def forward(self, x):
        """Generate embedding vectors for the given sequence of tokens.
        The final embeddings are computed by simply summing the word embeddings
        with the positional embeddings.

        Args:
            x: torch.Tensor
                Tensor of shape (B, T) giving a batch of sequences.

        Returns:
            embed: torch.Tensor
                Tensor of shape (B, T, D) giving the embeddings of each token.
        """
        _, T = x.shape
        pos = self.positions[:, :T]
        embed = self.pos_embed(pos) + self.word_embed(x) * self.scale
        return self.dropout(embed)

#