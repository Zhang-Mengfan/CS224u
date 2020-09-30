import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from bilstm_attn import SelfAttentiveModel
from utils import progress_bar


class SNLIBiLSTMAttentiveModel(nn.Module):
    def __init__(self, top_words=10000, emb_dim=300, hidden_size=64, num_classes=3,
                 max_seq_len=200, da=32, r=16, batch_size=500):
        super(SNLIBiLSTMAttentiveModel, self).__init__()

        self.top_words = top_words
        self.batch_size = batch_size
        self.embed_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.da = da
        self.r = r

        self.prem = SelfAttentiveModel(top_words=top_words,
                                       emb_dim=emb_dim,
                                       hidden_dim=hidden_size,
                                       max_seq_len=max_seq_len,
                                       da=da,
                                       r=r,
                                       batch_size=batch_size)

        self.hyp = SelfAttentiveModel(top_words=top_words,
                                      emb_dim=emb_dim,
                                      hidden_dim=hidden_size,
                                      max_seq_len=max_seq_len,
                                      da=da,
                                      r=r,
                                      batch_size=batch_size)

        self.output_layer = nn.Linear(4 * hidden_size, num_classes)

    def forward(self, x):
        x_prem, x_hyp = x
        encoded_prem = self.prem(x_prem)
        encoded_hyp = self.hyp(x_hyp)

        mult = encoded_prem.dot(encoded_hyp)
        diff = encoded_prem - encoded_hyp

        soft_in = torch.cat((encoded_prem, mult, diff, encoded_hyp), dim=1)
        logits = self.output_layer(soft_in)
        return logits

    def fit(self, x, y, **kwargs):
        """Standard `fit` method.

        Parameters
        ----------
        x : np.array
        y : array-like
        kwargs : dict
            For passing other parameters. If 'x_dev' is included,
            then performance is monitored every 10 epochs; use
            `dev_iter` to control this number.

        Returns
        -------
        self

        """
        # Incremental performance:
        x_dev = kwargs.get('x_dev')
        if x_dev is not None:
            dev_iter = kwargs.get('dev_iter', 10)

        # Data prep:
        x = np.array(x)
        self.input_dim = x.shape[1]
        self.classes_ = sorted(set(y))
        self.n_classes_ = len(self.classes_)
        class2index = dict(zip(self.classes_, range(self.n_classes_)))
        y = [class2index[label] for label in y]

        # Dataset:
        x = torch.tensor(x, dtype=float)
        y = torch.tensor(y, dtype=double)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            pin_memory=True)

        # Graph:
        self.model = self
        self.model.to(self.device)
        self.model.train()

        # Optimization:
        loss = nn.CrossEntropyLoss()
        optimizer = self.optimizer(
            self.model.parameters(),
            lr=self.eta,
            weight_decay=self.l2_strength)

        # Train:
        for iteration in range(1, self.max_iter+1):
            epoch_error = 0.0
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                batch_preds = self.model(x_batch)
                err = loss(batch_preds, y_batch)
                epoch_error += err.item()
                optimizer.zero_grad()
                err.backward()
                optimizer.step()
            # Incremental predictions where possible:
            if x_dev is not None and iteration > 0 and iteration % dev_iter == 0:
                self.dev_predictions[iteration] = self.predict(x_dev)
            self.errors.append(epoch_error)
            progress_bar("Finished epoch {} of {}; error is {}".format(iteration, self.max_iter, epoch_error))
        return self

    def predict_proba(self, x):
        """Predicted probabilities for the examples in `x`.

        Parameters
        ----------
        x : np.array

        Returns
        -------
        np.array with shape (len(x), self.n_classes_)

        """
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float).to(self.device)
            preds = self.model(x)
            return torch.softmax(preds, dim=1).cpu().numpy()

    def predict(self, x):
        """Predicted labels for the examples in `X`. These are converted
        from the integers that PyTorch needs back to their original
        values in `self.classes_`.

        Parameters
        ----------
        x : np.array

        Returns
        -------
        list of length len(X)

        """
        probs = self.predict_proba(x)
        return [self.classes_[i] for i in probs.argmax(axis=1)]
