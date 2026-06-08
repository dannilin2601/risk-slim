from adapt.utils import make_classification_da
import matplotlib.pyplot as plt


def plot_domains_2d(Xs, ys, Xt, yt,
                    source_marker="o",
                    target_marker="^",
                    figsize=(7, 5)):
    """
    Plot source and target domains in 2D.

    Parameters
    ----------
    Xs : ndarray, shape (n_source, 2)
        Source covariates.

    ys : ndarray, shape (n_source,)
        Source labels.

    Xt : ndarray, shape (n_target, 2)
        Target covariates.

    yt : ndarray, shape (n_target,)
        Target labels.

    source_marker : str, default='o'
        Marker for source samples.

    target_marker : str, default='^'
        Marker for target samples.

    figsize : tuple, default=(7, 5)
        Figure size.
    """

    if Xs.shape[1] != 2 or Xt.shape[1] != 2:
        raise ValueError("Both Xs and Xt must have exactly 2 features.")

    plt.figure(figsize=figsize)

    plt.scatter(
        Xs[:, 0],
        Xs[:, 1],
        c=ys,
        marker=source_marker,
        alpha=0.7,
        label="Source"
    )

    plt.scatter(
        Xt[:, 0],
        Xt[:, 1],
        c=yt,
        marker=target_marker,
        alpha=0.7,
        label="Target"
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Source and Target Domains")
    plt.legend()
    plt.colorbar(label="Class")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

Xs, ys, Xt, yt = make_classification_da(n_samples=100, 
                                        n_features=2,
                                        random_state=2)

print('Source domain shape:', Xs.shape)
print('Target domain shape:', Xt.shape)

plot_domains_2d(Xs, ys, Xt, yt)