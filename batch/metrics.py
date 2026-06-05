import pickle
import numpy as np
import pandas as pd

DATA = "examples/data/mammo_data.csv"
CV = "examples/data/mammo_cvindices.csv"
RESULTS = "batch/results/mammo_fold_{}_results.p"

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def auc_score(y, score):
    y = np.asarray(y)
    score = np.asarray(score)

    pos = score[y == 1]
    neg = score[y == 0]

    count = 0.0
    total = len(pos) * len(neg)

    for p in pos:
        count += np.sum(p > neg)
        count += 0.5 * np.sum(p == neg)

    return count / total

def calibration_error(y, p):
    # Overall calibration error
    return abs(np.mean(y) - np.mean(p))

df = pd.read_csv(DATA)
folds = pd.read_csv(CV, header=None).iloc[:, 0].values

y = df.iloc[:, 0].values
y = np.where(y == -1, 0, y)

X = df.iloc[:, 1:].values
X = np.column_stack([np.ones(X.shape[0]), X])

rows = []

for fold in range(1, 6):
    r = pickle.load(open(RESULTS.format(fold), "rb"))
    beta = r["solution"]

    test_idx = folds == fold

    scores = X[test_idx] @ beta
    probs = sigmoid(scores)

    auc = auc_score(y[test_idx], probs)
    cal = calibration_error(y[test_idx], probs)

    rows.append({
        "fold": fold,
        "auc": auc,
        "cal": cal,
        "loss": r["loss_value"],
        "gap": r["optimality_gap"],
        "model_size": np.sum(np.abs(beta[1:]) > 1e-9)
    })

res = pd.DataFrame(rows)

print(res)
print()
print("Mean test AUC:", res["auc"].mean())
print("Mean test CAL:", res["cal"].mean())
print("Mean test CAL (%):", 100 * res["cal"].mean())

# final fold-0 model
r0 = pickle.load(open(RESULTS.format(0), "rb"))
beta0 = r0["solution"]

print()
print("Final model fit using all data:")
print("Loss:", r0["loss_value"])
print("Optimality gap:", r0["optimality_gap"])
print("Model size:", np.sum(np.abs(beta0[1:]) > 1e-9))
print("Solution:", beta0)