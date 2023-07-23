from collections import Counter
import random

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from tqdm import tqdm

from transformer import Transformer


# Set the random seeds.
random.seed(0)
torch.manual_seed(seed=0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

# Get the Multi30k dataset.
train_data = Multi30k(root="datasets", split="train")
train_data = [(src, tgt) for src, tgt in train_data if len(src) > 0]

# Build src and tgt vocabs from the training set.
UNK, PAD, BOS, EOS = ("<UNK>", "<PAD>", "<START>", "<END>")
tokenizer = get_tokenizer("basic_english")
en_counter, de_counter = Counter(), Counter()
for src, tgt in train_data:
    en_counter.update(tokenizer(src))
    de_counter.update(tokenizer(tgt))
de_vocab = vocab(en_counter, specials=[UNK, PAD, BOS, EOS])
de_vocab.set_default_index(de_vocab[UNK])
en_vocab = vocab(de_counter, specials=[UNK, PAD, BOS, EOS])
en_vocab.set_default_index(en_vocab[UNK])
pad_idx = de_vocab[PAD] # pad_idx is 1
assert en_vocab[PAD] == de_vocab[PAD]


# Define a batch sampler.
class BatchSampler:
    """BatchSampler generates batches with minimum padding."""

    def __init__(self, lengths, batch_size):
        """Init a Batch Sampler.

        Args:
            lengths: List[int]
                A list containing the lengths of each source sequence in the
                dataset.
            batch_size: int
                The size of the batches yielded from the batch sampler.
        """
        self.lengths = lengths
        self.batch_size = batch_size

    def __iter__(self):
        """When iterating the batch sampler will sample a large pool of
        sequences, and will group them into batches, such that there is minimum
        padding required.
        """
        size = len(self.lengths)
        indices = list(range(size))
        random.shuffle(indices)

        # At each step sample a pool of sequence indices and sort by sequence
        # length. Once sorted, we can now sample batches, and each batch will
        # contain sequences of similar length.
        step = 100 * self.batch_size
        for i in range(0, size, step):
            pool = indices[i:i+step]
            pool = sorted(pool, key=lambda x: self.lengths[x])
            for j in range(0, len(pool), self.batch_size):
                if j + self.batch_size > len(pool): # assume drop_last=True
                    break
                # Ideally, there should also be some shuffling here.
                yield pool[j:j+self.batch_size]

    # Providing the __len__ method allows us to call `len(DataLoader)`.
    def __len__(self): return len(self.lengths) // self.batch_size


# Create the data loader.
lengths = [len(src) for src, _ in train_data]
batch_size = 128
train_loader = data.DataLoader(
    dataset=train_data,
    # batch_size=batch_size, shuffle=True, drop_last=True,
    batch_sampler=BatchSampler(lengths, batch_size),
    collate_fn=lambda batch: (
        pad_sequence(
            [torch.LongTensor(de_vocab(tokenizer(x))) for x, _ in batch],
            batch_first=True, padding_value=pad_idx),
        pad_sequence(
            [torch.LongTensor(en_vocab([BOS] + tokenizer(y) + [EOS])) for _, y in batch],
            batch_first=True, padding_value=pad_idx),
    ),
    num_workers=4,
)

# Define the model and the optimizer.
transformer = Transformer(
    src_vocab_size=len(de_vocab), tgt_vocab_size=len(en_vocab), max_seq_len=256,
    d_model=256, n_heads=8, n_enc=4, n_dec=4, dim_mlp=512, dropout=0.1,
)
transformer.to(device)
optim = torch.optim.AdamW(transformer.parameters(), lr=1e-4, weight_decay=1e-4)

# Train the model.
epochs = 30
pbar = tqdm(total=epochs*len(train_loader), desc="Iteration")
for e in range(epochs):
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)

        # Forward pass.
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        src_mask = (src != pad_idx)
        logits = transformer(src, tgt_in, src_mask)
        loss = F.cross_entropy(
            logits.permute(0,2,1), tgt_out, ignore_index=pad_idx,
        )

        # Back-prop.
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.)
        optim.step()

        # Update the progress bar.
        pbar.update()

pbar.close()

#