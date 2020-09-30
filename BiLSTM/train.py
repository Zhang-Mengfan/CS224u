from collections import Counter
# import os
# import argparse
# import pickle
import torch
# import json


from torch_rnn_classifier import TorchRNNClassifier
from full_model import SNLIBiLSTMAttentiveModel
import nli


SNLI_HOME = "/Users/bszalapski/Documents/StanfordCourses/CS224U/cs224u_project/cs224uSNLI/data/nlidata/snli_1.0"
BATCH_SIZE = 32
HIDDEN_SIZE = 300


class TorchRNNSentenceEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, seq_lengths, y):
        self.prem_seqs, self.hyp_seqs = sequences
        self.prem_lengths, self.hyp_lengths = seq_lengths
        self.y = y
        assert len(self.prem_seqs) == len(self.y)

    @staticmethod
    def collate_fn(batch):
        x_prem, x_hyp, prem_lengths, hyp_lengths, y = zip(*batch)
        prem_lengths = torch.LongTensor(prem_lengths)
        hyp_lengths = torch.LongTensor(hyp_lengths)
        y = torch.LongTensor(y)
        return (x_prem, x_hyp), (prem_lengths, hyp_lengths), y

    def __len__(self):
        return len(self.prem_seqs)

    def __getitem__(self, idx):
        return (self.prem_seqs[idx], self.hyp_seqs[idx],
                self.prem_lengths[idx], self.hyp_lengths[idx],
                self.y[idx])


class TorchBiLSTMAttentionClassifier(TorchRNNClassifier):

    def build_dataset(self, X, y):
        x_prem, x_hyp = zip(*X)
        x_prem, prem_lengths = self._prepare_dataset(x_prem)
        x_hyp, hyp_lengths = self._prepare_dataset(x_hyp)
        return TorchRNNSentenceEncoderDataset(
            (x_prem, x_hyp), (prem_lengths, hyp_lengths), y)

    def build_graph(self):
        return SNLIBiLSTMAttentiveModel(num_classes=3, batch_size=BATCH_SIZE, hidden_size=HIDDEN_SIZE)

    def predict_proba(self, X):
        with torch.no_grad():
            x_prem, x_hyp = zip(*X)
            x_prem, prem_lengths = self._prepare_dataset(x_prem)
            x_hyp, hyp_lengths = self._prepare_dataset(x_hyp)
            preds = self.model((x_prem, x_hyp), (prem_lengths, hyp_lengths))
            preds = torch.softmax(preds, dim=1).cpu().numpy()
            return preds


def sentence_encoding_rnn_phi(t1, t2):
    """Map `t1` and `t2` to a pair of lists of leaf nodes."""
    return (t1.leaves(), t2.leaves())


def get_sentence_encoding_vocab(X, n_words=None):
    wc = Counter([w for pair in X for ex in pair for w in ex])
    wc = wc.most_common(n_words) if n_words else wc.items()
    vocab = {w for w, c in wc}
    vocab.add("$UNK")
    return sorted(vocab)


def fit_bilstm_attention(X, y):
    vocab = get_sentence_encoding_vocab(X, n_words=10000)
    num_classes = 3
    batch_size = BATCH_SIZE
    hidden_size = HIDDEN_SIZE
    mod = SNLIBiLSTMAttentiveModel(num_classes=num_classes, batch_size=batch_size, hidden_size=hidden_size)
    mod.fit(X, y)
    return mod

# hidden_size=HIDDEN_SIZE,
#          num_classes=3,
#          epochs=64,
#          batch_size=BATCH_SIZE,
#          lr=0.0004,
#          patience=5,
#          checkpoint=None
def main():
    exp = nli.experiment(
        train_reader=nli.SNLITrainReader(SNLI_HOME, samp_percentage=1.0),
        assess_reader=nli.SNLIDevReader(SNLI_HOME, samp_percentage=1.0),
        phi=sentence_encoding_rnn_phi,
        train_func=fit_bilstm_attention,
        random_state=None,
        vectorize=False
    )
    print(exp)


if __name__ == '__main__':
    main()
