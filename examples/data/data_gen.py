from adapt.utils import make_classification_da

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

def plot_domains_2d(
    Xs, ys, Xt, yt,
    source_marker="o",
    target_marker="^",
    figsize=(7, 5),
    save_dir="synthetic_data",
    filename="source_target_domains.png",
    random_state=42
):

    if Xs.shape[1] != Xt.shape[1]:
        raise ValueError("Xs and Xt must have the same number of features.")

    n_features = Xs.shape[1]

    if n_features == 2:
        Xs_2d = Xs
        Xt_2d = Xt
        title = "Source and Target Domains"
        xlabel = "Feature 1"
        ylabel = "Feature 2"

    elif n_features > 2:
        X_all = np.vstack([Xs, Xt])

        X_all_2d = TSNE(
            n_components=2,
            random_state=random_state,
            init="pca",
            learning_rate="auto"
        ).fit_transform(X_all)

        Xs_2d = X_all_2d[:len(Xs)]
        Xt_2d = X_all_2d[len(Xs):]

        title = "Source and Target Domains using t-SNE"
        xlabel = "t-SNE 1"
        ylabel = "t-SNE 2"

    else:
        raise ValueError("Xs and Xt must have at least 2 features.")

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    save_dir = os.path.join(script_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=figsize)

    plt.scatter(
        Xs_2d[:, 0], Xs_2d[:, 1],
        c=ys,
        marker=source_marker,
        alpha=0.7,
        label="Source"
    )

    plt.scatter(
        Xt_2d[:, 0], Xt_2d[:, 1],
        c=yt,
        marker=target_marker,
        alpha=0.7,
        label="Target"
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.colorbar(label="Class")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to: {save_path}")



def save_domains_to_csv(
    Xs, ys, Xt, yt,
    save_dir="synthetic_data",
    source_filename="source_data.csv",
    target_filename="target_data.csv"
):
    """
    Save source and target datasets to CSV files.
    """

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    save_dir = os.path.join(script_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    source_df = pd.DataFrame(
        Xs,
        columns=[f"feature_{i+1}" for i in range(Xs.shape[1])]
    )
    source_df.insert(0, "label", ys)

    target_df = pd.DataFrame(
        Xt,
        columns=[f"feature_{i+1}" for i in range(Xt.shape[1])]
    )
    target_df.insert(0, "label", yt)

    source_path = os.path.join(save_dir, source_filename)
    target_path = os.path.join(save_dir, target_filename)

    source_df.to_csv(source_path, index=False)
    target_df.to_csv(target_path, index=False)

    # print(f"Source data saved to: {source_path}")
    # print(f"Target data saved to: {target_path}")

    return source_path, target_path

Xs, ys, Xt, yt = make_classification_da(n_samples=200, 
                                        n_features=8,
                                        random_state=2)


save_domains_to_csv(
    Xs, ys,
    Xt, yt,
    save_dir="synthetic_data"
)

plot_domains_2d(
    Xs, ys, Xt, yt,
    filename="domain_shift_example.png"
)

print('Source domain shape:', Xs.shape)
print('Target domain shape:', Xt.shape)