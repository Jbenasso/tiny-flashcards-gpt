import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json

class FlashcardDataset(Dataset):
    def __init__(self, path):
        with open(path, "r") as f:
            self.data = json.load(f)
        self.samples = [(d["context"], d["flashcard"]) for d in self.data]

        self.chars = sorted(list(set("".join([c for s in self.samples for c in (s[0]+s[1])] ))))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

    def encode(self, s): return [self.stoi[c] for c in s if c in self.stoi]
    def decode(self, l): return "".join([self.itos[i] for i in l])

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        ctx, card = self.samples[idx]
        x = torch.tensor(self.encode(ctx), dtype=torch.long)
        y = torch.tensor(self.encode(card), dtype=torch.long)
        return x, y

class TinyFlashcardLLM(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_heads=4, n_layers=4, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=n_heads, dim_feedforward=4*n_embd, activation="gelu"
            ) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.block_size, "Sequence too long!"
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.head(x)

def train():
    dataset = FlashcardDataset("data/flashcards.json")
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda b: (
        nn.utils.rnn.pad_sequence([x for x, _ in b], batch_first=True),
        nn.utils.rnn.pad_sequence([y for _, y in b], batch_first=True)
    ))

    model = TinyFlashcardLLM(dataset.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(5):
        for x, y in loader:
            logits = model(x)
            logits = logits[:, :y.size(1), :]
            loss = F.cross_entropy(logits.reshape(-1, dataset.vocab_size), y.reshape(-1), ignore_index=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, loss: {loss.item()}")

    torch.save(model.state_dict(), "tinyflashcards.pth")

if __name__ == "__main__":
    train()
