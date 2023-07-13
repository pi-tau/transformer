import torch
import torch.nn as nn

from encoder import EncoderBlock
from decoder import DecoderBlock
from embedding import EmbeddingLayer


class Transformer(nn.Module):
    """Transformer model as described in the paper "Attention is all you need".
    https://arxiv.org/abs/1706.03762
    """

    def __init__(
            self, src_vocab_size, tgt_vocab_size,
            d_model, n_heads, n_enc, n_dec, dim_mlp=2048, dropout=0.0,
        ):
        """Init a Transformer model.

        Args:
            src_vocab_size: int
                Size of the source vocab.
            tgt_vocab_size: int
                Size of the target vocab.
            d_model: int
                Size of the encoder and decoder layers. Because the model uses
                residual connections, all encodings will have the same size.
            n_heads: int
                Number of heads for multi-head attention.
            n_enc: int
                Number of Encoder layers in the Encoder.
            n_dec: int
                Number of Decoder layers in the Decoder.
            dim_mlp: int, optional
                Dimension of the hidden layer of the MLP network. Default: 2048.
            dropout: int, optional
                Dropout rate applied for encoder, decoder and embedding layers.
                Default: 0.
        """
        super().__init__()

        # Define the embedding layers for the src and tgt sequences.
        self.src_embed = EmbeddingLayer(src_vocab_size, d_model)
        self.tgt_embed = EmbeddingLayer(tgt_vocab_size, d_model)

        # Define the encoder and the decoder modules.
        # Note that because of the Pre-LN architecture we need to apply layer
        # norm to the outputs of the encoder and decoder stacks.
        self.encoder_stack = nn.ModuleList((
            EncoderBlock(d_model, n_heads, dim_mlp, dropout) for _ in range(n_enc)
        ))
        self.enc_norm = nn.LayerNorm(d_model)
        self.decoder_stack = nn.ModuleList((
            DecoderBlock(d_model, n_heads, dim_mlp, dropout) for _ in range(n_dec)
        ))
        self.dec_norm = nn.LayerNorm(d_model)

        # Project the embeddings from the decoder into the target vocab space.
        self.tgt_vocab_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Initialize model parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        z = self.src_embed(src)
        for encoder in self.encoder_stack:
            z = encoder(z, src_mask)
        return self.enc_norm(z)

    def decode(self, tgt, mem, mem_mask):
        z = self.tgt_embed(tgt)
        for decoder in self.decoder_stack:
            z = decoder(z, mem, mem_mask)
        return self.dec_norm(z)

    def forward(self, src, tgt, src_mask=None):
        """Given a source sequence, perform a forward pass through the
        transformer and return the logit scores for the generated decodings.
        The model uses teacher forcing and during decoding feeds the elements of
        the target sequence.

        Args:
            src: torch.Tensor
                Torch tensor of shape (B, T_src), giving a batch of padded
                source sequences.
            tgt: torch.Tensor
                Torch tensor of shape (B, T_tgt), giving a batch of
                corresponding padded target sequences.
            src_mask: torch.Tensor, optional
                Boolean tensor of shape (B, T_src) indicating  which elements of
                the src sequence should be masked. Usually used for masking out
                the padded part. A value of True indicates that the element
                *should* take part in the computation. Default: None.

        Returns:
            tgt_log_probs: torch.Tensor
                Torch tensor of shape (B, T_tgt, tgt_vocab) assigning logit
                scores over the target vocab for each element of the target
                sequence.
        """
        mem = self.encode(src, src_mask)
        out = self.decode(tgt, mem, src_mask)
        tgt_scores = self.tgt_vocab_proj(out)
        return tgt_scores

    @torch.no_grad()
    def greedy_decode(self, src, src_mask, bos_idx, eos_idx, max_len=80):
        """Greedy decoding of the source sequences.

        Args:
            src: torch.Tensor
                Torch tensor of shape (B, T_src), giving a batch of padded
                source sequences
            src_mask: torch.Tensor
                Boolean tensor of shape (B, T_src) indicating  which elements of
                the src sequence should be masked. Usually used for masking out
                the padded part. A value of True indicates that the element
                *should* take part in the computation. Pass None for no masking.
            bos_idx: int
                The index of the token for beginning the sequence.
            eos_idx: int
                The index of the token for ending the sequence.
            max_len: int, optional
                Maximum number of tokens in the decoded sequence. Default: 80.

        Returns:
            tgt: torch.Tensor
                Torch tensor of shape (B, T), giving a batch of decodings.
        """
        B = src.shape[0]
        done = {i : False for i in range(B)}
        was_training = self.training
        self.eval()

        # The initial decoding consists only of the <START> token. At each step
        # of the decoding process we will extend the decoded sequence.
        tgt = torch.LongTensor([[bos_idx]] * B).to(src.device)

        mem = self.encode(src, src_mask)
        for _ in range(max_len-1):
            # Decode the next element of the target sequence.
            out = self.decode(tgt, mem, mem_mask=src_mask)
            scores = self.tgt_vocab_proj(out)
            next_idx = torch.max(scores[:, -1:], dim=-1).indices
            tgt = torch.concat((tgt, next_idx), dim=1)

            # Keep track of the number of decoded sequences.
            for i, idx in enumerate(next_idx):
                if idx[0] == eos_idx: done[i] = True
            if False not in done.values(): break

        if was_training: self.train()

        return tgt


if __name__ == "__main__":
    # Train the Transformer model to reverse a sequence of given inputs.
    # RNNs can have issues with such tasks because this requires long-term
    # dependencies. We expect the Transformer to perform well because of the
    # all-to-all self-attention mechanism.
    from random import seed, randint
    import torch.nn.functional as F
    import torch.utils.data as data
    from torch.nn.utils.rnn import pad_sequence
    from tqdm import tqdm

    seed(0)
    torch.manual_seed(seed=0)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bos_idx, eos_idx, pad_idx = 1, 2, 0
    vocab_size, max_len = 100, 32

    data_loader = data.DataLoader(  # random sequences of different lengths
        dataset=[torch.randint(3, vocab_size, (randint(max_len//2, max_len),)) for _ in range(50000)],
        batch_size=128, shuffle=True, drop_last=True,
        collate_fn=lambda batch: (
            pad_sequence(batch, batch_first=True, padding_value=pad_idx),
            pad_sequence(           # flip the sequence and add <START> and <END> tags
                [torch.LongTensor([bos_idx] + x.flip(0).tolist() + [eos_idx]) for x in batch],
                batch_first=True, padding_value=pad_idx,
        )),
    )

    transformer = Transformer(
        src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
        d_model=64, n_heads=1, n_enc=2, n_dec=2, dim_mlp=128, dropout=0.3,
    ).to(device)
    optim = torch.optim.AdamW(transformer.parameters(), lr=1e-3, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda i: i/1000 if i <= 1000 else (i/1000)**(-0.5)
    )                               # linear warm-up        # quadratic slow-down

    epochs = 50
    pbar = tqdm(total=epochs*len(data_loader), desc="Iteration")
    for e in range(epochs):
        epoch_loss = 0.
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = transformer(src, tgt_in, (src != pad_idx))
            loss = F.cross_entropy(
                logits.permute(0,2,1), tgt_out, ignore_index=pad_idx, reduction="mean")
            epoch_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.)
            optim.step()
            lr_scheduler.step()

            pbar.update()

        x1 = torch.LongTensor([3, 5, 8, 13, 21, 34, 55]).unsqueeze(dim=0).to(device)
        y1 = transformer.greedy_decode(x1, None, bos_idx, eos_idx)
        x2 = torch.randint(low=3, high=vocab_size, size=(15,)).unsqueeze(dim=0).to(device)
        y2 = transformer.greedy_decode(x2, None, bos_idx, eos_idx)
        tqdm.write(f"Epoch: {e+1}\n\tx1: {x1}\n\ty1: {y1}\n\n\tx2: {x2}\n\ty2: {y2}")
        tqdm.write(f"\tloss: {epoch_loss/len(data_loader):.5f}")

    pbar.close()

#

"""
>>> a, b = next(iter(data_loader))
>>> src, tgt = a.to(device), b.to(device)
>>> tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
>>> mask = (src != pad_idx)
>>> mem = transformer.encode(src, mask)
>>> x = transformer.tgt_embed(tgt_in)
>>> _, T, _ = x.shape
>>> causal_mask = torch.ones(1, T, T, dtype=torch.bool).tril().to(device)
>>> z, attn = transformer.decoder_stack[0].self_attn(x,x,x,mask=causal_mask, need_attn=True)
>>> attn[0][0]
"""