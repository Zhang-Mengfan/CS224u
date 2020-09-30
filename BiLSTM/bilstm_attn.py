import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pack_padded_sequence


# Adapted from: https://github.com/PrashantRanjan09/Structured-Self-Attentive-Sentence-Embedding
class SelfAttentiveModel(nn.Module):

    def __init__(self, top_words=10000, emb_dim=300, hidden_dim=64, max_seq_len=200, da=32, r=16, batch_size=500):
        super(SelfAttentiveModel, self).__init__()
        self.top_words = top_words
        self.batch_size = batch_size
        self.embed_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.da = da
        self.r = r

        self.embedding = nn.Embedding(top_words, emb_dim)
        self.bilstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.lin1 = nn.Linear(2 * hidden_dim, da)
        self.lin2 = nn.Linear(da, r)
        self.lin3 = nn.Linear(r * 2 * hidden_dim, 1)

    def forward(self, x):
        out = self.embedding(x)
        out_lstm, _ = self.bilstm(out)
        out = self.lin1(out_lstm)
        out = F.tanh(out)
        out = self.lin2(out)
        out_a = F.softmax(out, dim=0)
        temp1 = out_a.permute(0, 2, 1)
        temp2 = out_lstm
        out = torch.bmm(temp1, temp2)  # AH
        out = out.view(self.batch_size, 2 * self.r * self.hidden_dim)
        out = self.lin3(out)
        out = F.sigmoid(out)

        return out
