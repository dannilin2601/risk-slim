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


def build_scorecard(beta, feature_names):

    rows = []

    for name, coef in zip(feature_names, beta[1:]):

        if coef != 0:
            rows.append({
                "feature": name,
                "points": int(coef)
            })

    return pd.DataFrame(rows)


def build_risk_table(beta):
    """
    Build score-to-risk lookup table.
    """

    intercept = beta[0]
    coefs = beta[1:]

    min_score = int(np.sum(coefs[coefs < 0]))
    max_score = int(np.sum(coefs[coefs > 0]))

    rows = []

    for score in range(min_score, max_score + 1):

        risk = sigmoid(intercept + score)

        rows.append({
            "score": score,
            "risk": risk,
            "risk_percent": 100 * risk
        })

    return pd.DataFrame(rows)

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
print("Mean test AUC:", np.round(res["auc"].mean(), 3))
print("Mean test CAL:", np.round(res["cal"].mean(), 3))
print("Mean test CAL (%):", np.round(100 * res["cal"].mean(), 1))

# final fold-0 model
r0 = pickle.load(open(RESULTS.format(0), "rb"))
beta0 = r0["solution"]

print()
print("Final model fit using all data:")
print("Loss:", np.round(r0["loss_value"], 3))
print("Optimality gap (%):", np.round(r0["optimality_gap"], 3))
print("Model size:", np.sum(np.abs(beta0[1:]) > 1e-9))
print("Solution:", beta0)

print()
r0 = pickle.load(open(RESULTS.format(0), "rb"))
beta0 = r0["solution"]

scorecard = build_scorecard(
    beta0,
    df.columns[1:]
)

risk_table = build_risk_table(
    beta0
)

print()
print("Fold-0 Scorecard:")
print(scorecard.to_string(index=False))

print()
print("Risk Table:")
print(
    risk_table.to_string(
        index=False,
        formatters={
            "risk": "{:.4f}".format,
            "risk_percent": "{:.1f}%".format
        }
    )
)