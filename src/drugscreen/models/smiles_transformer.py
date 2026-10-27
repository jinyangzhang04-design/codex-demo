"""A compact SMILES Transformer for reproducible domain pretraining."""
from __future__ import annotations
import re
import torch
from torch import nn

TOKEN_RE = re.compile(r"\[[^\]]+\]|Br|Cl|Si|Na|Li|Al|Ca|Fe|Mg|Zn|[A-Za-z]|\d|[^\s]")

def split_smiles(smiles: str) -> list[str]:
    return TOKEN_RE.findall(smiles)

class SmilesTokenizer:
    special = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"]

    def __init__(self, vocabulary: list[str], max_length: int = 96):
        self.vocabulary = self.special + [token for token in vocabulary if token not in self.special]
        self.token_to_id = {token: i for i, token in enumerate(self.vocabulary)}
        self.max_length = max_length
        self.pad_id, self.unk_id, self.bos_id, self.eos_id, self.mask_id = range(5)

    @classmethod
    def build(cls, smiles_list, max_length=96):
        tokens = sorted({token for smiles in smiles_list for token in split_smiles(smiles)})
        return cls(tokens, max_length)

    def encode(self, smiles: str) -> list[int]:
        values = [self.bos_id] + [self.token_to_id.get(token, self.unk_id) for token in split_smiles(smiles)]
        values = values[: self.max_length - 1] + [self.eos_id]
        return values + [self.pad_id] * (self.max_length - len(values))

    def state(self):
        return {"vocabulary": self.vocabulary, "max_length": self.max_length}

    @classmethod
    def from_state(cls, state):
        return cls(state["vocabulary"][5:], state["max_length"])

class SmilesEncoder(nn.Module):
    def __init__(self, vocabulary_size, max_length=96, d_model=128, nhead=4, layers=2, feedforward=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        block = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=feedforward, dropout=0.1, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.max_length = max_length
        self.d_model = d_model

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        padding = input_ids.eq(0)
        hidden = self.transformer(hidden, src_key_padding_mask=padding)
        hidden = self.norm(hidden)
        valid = (~padding).unsqueeze(-1).float()
        return (hidden * valid).sum(1) / valid.sum(1).clamp_min(1.0)

class SmilesMLM(nn.Module):
    def __init__(self, vocabulary_size, max_length=96, d_model=128, nhead=4, layers=2, feedforward=256):
        super().__init__()
        self.encoder = SmilesEncoder(vocabulary_size, max_length, d_model, nhead, layers, feedforward)
        self.lm_head = nn.Linear(d_model, vocabulary_size)

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.encoder.token_embedding(input_ids) + self.encoder.position_embedding(positions)
        padding = input_ids.eq(0)
        hidden = self.encoder.transformer(hidden, src_key_padding_mask=padding)
        return self.lm_head(self.encoder.norm(hidden))

class SmilesMultiTask(nn.Module):
    def __init__(self, encoder: SmilesEncoder, task_count: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(encoder.d_model, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, task_count))

    def forward(self, input_ids):
        return self.head(self.encoder(input_ids))
