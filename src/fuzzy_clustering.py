import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import skfuzzy as fuzz
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from loguru import logger
import joblib

N_CLUSTERS = 15
FUZZINESS_M = 1.5
PCA_COMPONENTS = 50
FCM_ERROR = 0.005
FCM_MAX_ITER = 300
DOMINANT_THRESHOLD = 0.25
BOUNDARY_THRESHOLD = 0.2

def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = PCA_COMPONENTS,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA]:
    
    logger.info(f"Reducing {embeddings.shape[1]}-dim embeddings to {n_components}-dim with PCA")
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)

    explained = np.sum(pca.explained_variance_ratio_)
    logger.info(f"PCA explains {explained:.1%} of variance with {n_components} components")

    reduced = normalize(reduced, norm="l2")

    return reduced, pca

def select_k_with_elbow(
    reduced_embeddings: np.ndarray,
    k_range: range = range(8, 25),
    fuzziness: float = FUZZINESS_M,
    output_dir: Optional[str] = None,
) -> dict:
   
    logger.info(f"Running cluster selection analysis over k={list(k_range)}")
    results = {}


    data_T = reduced_embeddings.T

    for k in tqdm(k_range, desc="Testing k values"):

        cntr, u, _, d, jm, p, fpc = fuzz.cluster.cmeans(
            data=data_T,
            c=k,
            m=fuzziness,
            error=FCM_ERROR,
            maxiter=FCM_MAX_ITER,
            init=None,
            seed=42,
        )

       
        hard_labels = np.argmax(u, axis=0)

       
        eps = 1e-10  
        partition_entropy = -np.mean(np.sum(u * np.log(u + eps), axis=0))


        unique_labels = np.unique(hard_labels)
        if len(unique_labels) >= 2 and len(hard_labels) > k:

            sample_size = min(5000, len(hard_labels))
            sample_idx = np.random.choice(len(hard_labels), sample_size, replace=False)
            sil = silhouette_score(
                reduced_embeddings[sample_idx],
                hard_labels[sample_idx],
                metric="cosine",
            )
        else:
            sil = -1.0

        results[k] = {
            "objective_final": float(jm[-1]),
            "silhouette": float(sil),
            "partition_entropy": float(partition_entropy),
            "fpc": float(fpc),  
            "iterations": int(p),
        }

        logger.info(
            f"  k={k:2d}: objective={jm[-1]:.2f}, silhouette={sil:.4f}, "
            f"PE={partition_entropy:.4f}, FPC={fpc:.4f}"
        )


    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _plot_k_selection(results, output_dir)

    return results

def _plot_k_selection(results: dict, output_dir: str) -> None:

    ks = sorted(results.keys())
    objectives = [results[k]["objective_final"] for k in ks]
    silhouettes = [results[k]["silhouette"] for k in ks]
    entropies = [results[k]["partition_entropy"] for k in ks]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(ks, objectives, "b-o", markersize=4)
    axes[0].axvline(x=N_CLUSTERS, color="red", linestyle="--", label=f"k={N_CLUSTERS}")
    axes[0].set_title("FCM Objective (Elbow)")
    axes[0].set_xlabel("Number of Clusters k")
    axes[0].set_ylabel("Objective Function")
    axes[0].legend()

    axes[1].plot(ks, silhouettes, "g-o", markersize=4)
    axes[1].axvline(x=N_CLUSTERS, color="red", linestyle="--")
    axes[1].set_title("Silhouette Score (Hard Labels)")
    axes[1].set_xlabel("Number of Clusters k")
    axes[1].set_ylabel("Silhouette Score")

    axes[2].plot(ks, entropies, "r-o", markersize=4)
    axes[2].axvline(x=N_CLUSTERS, color="red", linestyle="--")
    axes[2].set_title("Partition Entropy")
    axes[2].set_xlabel("Number of Clusters k")
    axes[2].set_ylabel("Entropy (lower = crisper)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_selection_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved cluster selection plot")
    
def run_fcm(
    reduced_embeddings: np.ndarray,
    n_clusters: int = N_CLUSTERS,
    fuzziness: float = FUZZINESS_M,
) -> tuple[np.ndarray, np.ndarray]:
    
    logger.info(f"Running final FCM with k={n_clusters}, m={fuzziness}")


    data_T = reduced_embeddings.T

    cntr, u, _, _, jm, iterations, fpc = fuzz.cluster.cmeans(
        data=data_T,
        c=n_clusters,
        m=fuzziness,
        error=FCM_ERROR,
        maxiter=FCM_MAX_ITER,
        init=None,
        seed=42,
    )

    logger.info(
        f"FCM converged in {iterations} iterations. "
        f"Final objective: {jm[-1]:.4f}, FPC: {fpc:.4f}"
    )


    membership = u.T

    return cntr, membership

def analyze_clusters(
    membership: np.ndarray,
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> dict:
  
    n_docs, n_clusters = membership.shape


    hard_labels = np.argmax(membership, axis=1)


    eps = 1e-10
    per_doc_entropy = -np.sum(membership * np.log(membership + eps), axis=1)

    df = df.copy()
    df["dominant_cluster"] = hard_labels
    df["membership_entropy"] = per_doc_entropy
    df["max_membership"] = membership.max(axis=1)

    analysis = {}

    for cluster_id in range(n_clusters):

        core_mask = (hard_labels == cluster_id)
        core_docs = df[core_mask]


        boundary_mask = (
            (hard_labels == cluster_id) &
            (membership[:, cluster_id] > BOUNDARY_THRESHOLD) &
            (per_doc_entropy > np.percentile(per_doc_entropy, 75))
        )
        boundary_docs = df[boundary_mask]


        newsgroup_dist = core_docs["newsgroup"].value_counts().to_dict()


        core_samples = core_docs["clean_text"].head(3).apply(lambda x: x[:200]).tolist()


        boundary_samples = boundary_docs.head(3).apply(
            lambda row: {
                "text": row["clean_text"][:200],
                "newsgroup": row["newsgroup"],
                "membership_vector": membership[row.name, :].round(3).tolist(),
            },
            axis=1,
        ).tolist() if len(boundary_docs) > 0 else []

        analysis[cluster_id] = {
            "size": int(core_mask.sum()),
            "dominant_newsgroups": newsgroup_dist,
            "core_samples": core_samples,
            "boundary_samples": boundary_samples,
            "avg_membership": float(membership[core_mask, cluster_id].mean()) if core_mask.sum() > 0 else 0.0,
        }

        logger.info(
            f"Cluster {cluster_id:2d}: {core_mask.sum():4d} docs | "
            f"Top groups: {list(newsgroup_dist.keys())[:3]}"
        )


    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "cluster_analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info("Saved cluster analysis JSON")


        doc_assignments = df[["doc_id", "newsgroup", "dominant_cluster", "membership_entropy", "max_membership"]].copy()

        for c in range(n_clusters):
            doc_assignments[f"cluster_{c}_membership"] = membership[:, c]
        doc_assignments.to_parquet(os.path.join(output_dir, "doc_cluster_assignments.parquet"), index=False)
        logger.info("Saved per-document cluster assignments")

    return analysis

def visualize_clusters_2d(
    embeddings: np.ndarray,
    membership: np.ndarray,
    output_dir: str,
    sample_size: int = 5000,
) -> None:
    
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed, skipping 2D visualization")
        return

    logger.info("Generating UMAP 2D visualization...")


    n = min(sample_size, len(embeddings))
    idx = np.random.choice(len(embeddings), n, replace=False)
    sample_emb = embeddings[idx]
    sample_membership = membership[idx]
    hard_labels = np.argmax(sample_membership, axis=1)

    reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine", n_neighbors=15)
    coords_2d = reducer.fit_transform(sample_emb)

    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=hard_labels,
        cmap="tab20",
        alpha=0.4,
        s=5,
    )
    plt.colorbar(scatter, label="Cluster ID")
    plt.title(f"UMAP 2D Projection — Fuzzy Clusters (k={membership.shape[1]}, m={FUZZINESS_M})")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "umap_clusters.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved UMAP visualization")

def save_clustering_artifacts(
    pca_model: PCA,
    fcm_centers: np.ndarray,
    membership: np.ndarray,
    output_dir: str,
) -> None:

    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(pca_model, os.path.join(output_dir, "pca_model.joblib"))
    np.save(os.path.join(output_dir, "fcm_centers.npy"), fcm_centers)
    np.save(os.path.join(output_dir, "membership_matrix.npy"), membership)

    logger.info(f"Clustering artifacts saved to {output_dir}")
    logger.info(f"  PCA model: {output_dir}/pca_model.joblib")
    logger.info(f"  FCM centers: {output_dir}/fcm_centers.npy — shape {fcm_centers.shape}")
    logger.info(f"  Membership matrix: {output_dir}/membership_matrix.npy — shape {membership.shape}")


def load_clustering_artifacts(artifacts_dir: str) -> tuple[PCA, np.ndarray, np.ndarray]:

    pca_model = joblib.load(os.path.join(artifacts_dir, "pca_model.joblib"))
    fcm_centers = np.load(os.path.join(artifacts_dir, "fcm_centers.npy"))
    membership = np.load(os.path.join(artifacts_dir, "membership_matrix.npy"))
    return pca_model, fcm_centers, membership

def run_clustering(
    embeddings_path: str,
    corpus_meta_path: str,
    output_dir: str = "data",
    run_k_selection: bool = True,
) -> tuple[np.ndarray, np.ndarray]:


    embeddings = np.load(embeddings_path)
    df = pd.read_parquet(corpus_meta_path)

    logger.info(f"Loaded {len(embeddings)} embeddings, {len(df)} corpus documents")
    assert len(embeddings) == len(df), "Embeddings and corpus must have same length"


    reduced, pca_model = reduce_dimensions(embeddings)


    if run_k_selection:
        select_k_with_elbow(reduced, output_dir=output_dir)


    centers, membership = run_fcm(reduced)


    analyze_clusters(membership, df, output_dir=output_dir)


    visualize_clusters_2d(reduced, membership, output_dir=output_dir)


    save_clustering_artifacts(pca_model, centers, membership, output_dir)

    return centers, membership


if __name__ == "__main__":
    import sys
    embeddings_path = sys.argv[1] if len(sys.argv) > 1 else "data/embeddings.npy"
    corpus_path = sys.argv[2] if len(sys.argv) > 2 else "data/corpus_meta.parquet"
    run_clustering(embeddings_path, corpus_path)
