#"LSTM klasifikátor nad postupnosťou príznakov a jeho trénovanie.
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import CLASSES


class LSTMClassifier(nn.Module):
    def __init__(self, in_dim=512, hidden=128, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, len(CLASSES))

    def forward(self, x):
        _, (h, _) = self.lstm(x)        # h[-1]: skrytý stav po poslednom kroku
        return self.fc(self.drop(h[-1]))


def class_weights(y):
    counts = np.bincount(y, minlength=len(CLASSES))
    return torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)

#LSTM
def train_lstm(X_tr, y_tr, X_va, y_va, device, seed=0, epochs=40, lr=1e-3):
    torch.manual_seed(seed)
    model = LSTMClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_tr).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X_tr, torch.tensor(y_tr)), batch_size=32, shuffle=True)
    X_va, y_va = X_va.to(device), torch.tensor(y_va, device=device)

    history, best_loss, best_epoch, best_state = [], np.inf, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va), y_va).item()
        history.append((train_loss / len(y_tr), val_loss))

        if val_loss < best_loss:
            best_loss, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history, best_epoch
