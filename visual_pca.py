import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def visualize_3d_pca(
    normal_features,
    defect_features,
    normal_test_features=None,
    max_samples=5000,
    save_path=None,
):
    """
    Parameters
    ----------
    normal_features:
        train normal patch features
        shape = (N, C)

    defect_features:
        defect image patch features
        shape = (M, C)

    normal_test_features:
        optional test normal features
        shape = (K, C)
    """

    # --------------------------------------------------
    # Subsampling
    # --------------------------------------------------
    rng = np.random.default_rng(42)

    if len(normal_features) > max_samples:
        idx = rng.choice(
            len(normal_features),
            max_samples,
            replace=False
        )
        normal_features = normal_features[idx]

    if len(defect_features) > max_samples:
        idx = rng.choice(
            len(defect_features),
            max_samples,
            replace=False
        )
        defect_features = defect_features[idx]

    if (
        normal_test_features is not None
        and len(normal_test_features) > max_samples
    ):
        idx = rng.choice(
            len(normal_test_features),
            max_samples,
            replace=False
        )

        normal_test_features = normal_test_features[idx]


    # --------------------------------------------------
    # L2 normalization
    #
    # DINO / ADPretrain embedding 비교에서는 권장
    # --------------------------------------------------
    normal_features = normalize(
        normal_features,
        norm="l2"
    )

    defect_features = normalize(
        defect_features,
        norm="l2"
    )

    feature_list = [
        normal_features,
        defect_features,
    ]


    if normal_test_features is not None:

        normal_test_features = normalize(
            normal_test_features,
            norm="l2"
        )

        feature_list.append(
            normal_test_features
        )


    # --------------------------------------------------
    # PCA는 모든 feature를 함께 넣어서 fit
    # --------------------------------------------------
    all_features = np.concatenate(
        feature_list,
        axis=0
    )

    print(
        "Input feature shape:",
        all_features.shape
    )


    pca = PCA(
        n_components=3,
        random_state=42
    )

    pca_features = pca.fit_transform(
        all_features
    )


    print(
        "Explained variance ratio:",
        pca.explained_variance_ratio_
    )

    print(
        "Total explained variance:",
        pca.explained_variance_ratio_.sum()
    )


    # --------------------------------------------------
    # 다시 group별 분리
    # --------------------------------------------------
    n_normal = len(normal_features)
    n_defect = len(defect_features)

    normal_pca = pca_features[
        :n_normal
    ]

    defect_pca = pca_features[
        n_normal:
        n_normal + n_defect
    ]


    if normal_test_features is not None:

        normal_test_pca = pca_features[
            n_normal + n_defect:
        ]


    # --------------------------------------------------
    # 3D plot
    # --------------------------------------------------
    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )


    ax.scatter(
        normal_pca[:, 0],
        normal_pca[:, 1],
        normal_pca[:, 2],
        s=8,
        alpha=0.3,
        label="Train Normal",
    )


    if normal_test_features is not None:

        ax.scatter(
            normal_test_pca[:, 0],
            normal_test_pca[:, 1],
            normal_test_pca[:, 2],
            s=12,
            alpha=0.5,
            label="Test Normal",
        )


    ax.scatter(
        defect_pca[:, 0],
        defect_pca[:, 1],
        defect_pca[:, 2],
        s=20,
        alpha=0.8,
        label="Defect",
    )


    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
    )

    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
    )

    ax.set_zlabel(
        f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)"
    )

    ax.set_title(
        "3D PCA of ADPretrain Patch Features"
    )

    ax.legend()

    plt.tight_layout()


    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )


    plt.show()