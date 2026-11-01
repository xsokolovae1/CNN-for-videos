from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import CLASSES, N_SPLITS, RESULTS_DIR, TEST_SIZE
from .models import train_lstm


# delenie dát
def make_splits(df, n_splits=N_SPLITS):

    for split in range(n_splits):
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=split)
        tr, te = next(gss.split(df, groups=df.source))
        yield split, tr, te


def train_val_split(df, tr, split):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=split)
    a, b = next(gss.split(tr, groups=df.source.values[tr]))
    return tr[a], tr[b]


def predictions_table(split, te, y_true, y_pred):
    return pd.DataFrame({"split": split, "clip": te, "y_true": y_true, "y_pred": y_pred})


# modely

def svm_search(X, y, groups):
    grid = [
        {"svm__kernel": ["linear"], "svm__C": [0.001, 0.01, 0.1, 1]},
        {"svm__kernel": ["rbf"], "svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", 1e-3, 1e-4]},
    ]
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("svm", SVC(class_weight="balanced"))])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy")
    search.fit(X, y, groups=groups)
    return search


def evaluate_svm(X, df):
    y, groups = df.label.values, df.source.values
    tables = []
    for split, tr, te in make_splits(df):
        search = svm_search(X[tr], y[tr], groups[tr])
        tables.append(predictions_table(split, te, y[te], search.predict(X[te])))
    return pd.concat(tables, ignore_index=True)


def evaluate_lstm(feats, df, device):
    y = df.label.values
    tables, first_history = [], None
    for split, tr, te in make_splits(df):
        i_tr, i_va = train_val_split(df, tr, split)

        # štandardizácia príznakov podľa trénovacích dát
        mean, std = feats[i_tr].mean(axis=(0, 1)), feats[i_tr].std(axis=(0, 1)) + 1e-6
        to_tensor = lambda idx: torch.tensor((feats[idx] - mean) / std, dtype=torch.float32)

        model, history, best_epoch = train_lstm(to_tensor(i_tr), y[i_tr], to_tensor(i_va), y[i_va],
                                                device, seed=split)
        model.eval()
        with torch.no_grad():
            pred = model(to_tensor(te).to(device)).argmax(1).cpu().numpy()
        tables.append(predictions_table(split, te, y[te], pred))
        if first_history is None:
            first_history = (history, best_epoch)
    return pd.concat(tables, ignore_index=True), first_history


def evaluate_majority(df):
    y = df.label.values
    tables = []
    for split, tr, te in make_splits(df):
        majority = np.bincount(y[tr]).argmax()
        tables.append(predictions_table(split, te, y[te], np.full(len(te), majority)))
    return pd.concat(tables, ignore_index=True)


# uloženie
def save_predictions(pred, name):
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    pred.to_csv(Path(RESULTS_DIR) / f"{name}.csv", index=False)


def load_predictions(name):
    return pd.read_csv(Path(RESULTS_DIR) / f"{name}.csv")


# metriky
def split_scores(pred):
    rows = []
    for split, g in pred.groupby("split"):
        correct = g.y_true == g.y_pred
        row = {"presnosť (accuracy)": correct.mean() * 100}
        for label, cls in enumerate(CLASSES):
            row[f"správne {cls} (recall)"] = correct[g.y_true == label].mean() * 100
        rows.append(row)
    return pd.DataFrame(rows)


def summary(pred, name):
    scores = split_scores(pred)
    row = {col: f"{scores[col].mean():.0f} % ± {scores[col].std():.0f} %" for col in scores}
    return pd.DataFrame([row], index=[name])


# grafy

def plot_confusion(pred, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, normalize, fmt, subtitle in [(axes[0], None, "d", "počet klipov"),
                                         (axes[1], "true", ".0%", "% klipov skutočnej triedy")]:
        ConfusionMatrixDisplay.from_predictions(
            pred.y_true, pred.y_pred, display_labels=CLASSES, normalize=normalize,
            values_format=fmt, cmap="Blues", colorbar=False, ax=ax)
        ax.set_xlabel("predpoveď modelu")
        ax.set_ylabel("skutočná trieda")
        ax.set_title(subtitle)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_history(history, best_epoch):
    history = np.array(history)
    epochs = range(1, len(history) + 1)
    plt.plot(epochs, history[:, 0], label="trénovacie dáta")
    plt.plot(epochs, history[:, 1], label="validačné dáta")
    plt.axvline(best_epoch, ls="--", c="gray", label="vybraná epocha")
    plt.xlabel("epocha")
    plt.ylabel("strata (cross-entropy loss)")
    plt.legend()
    plt.show()
