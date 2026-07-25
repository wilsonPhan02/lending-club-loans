"""
Modul 04 — Dimensionality Reduction.
PCA (>=90% varians) -> input K-Means/Hierarchical.
UMAP 2D (sample) -> input DBSCAN.
Mengembalikan (df_pca, df_umap, pca).
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import config as cfg


def run_pca(df_final_scaled: pd.DataFrame, verbose: bool = True):
    log = print if verbose else (lambda *a, **k: None)
    cum = np.cumsum(PCA().fit(df_final_scaled).explained_variance_ratio_)
    n_comp = int(np.argmax(cum >= cfg.PCA_VARIANCE) + 1)
    pca = PCA(n_components=n_comp, random_state=cfg.RANDOM_SEED)
    X = pca.fit_transform(df_final_scaled)
    df_pca = pd.DataFrame(X, columns=[f"PC{i+1}" for i in range(n_comp)],
                          index=df_final_scaled.index)
    log(f"[reduce] PCA: {n_comp} komponen ({cfg.PCA_VARIANCE:.0%} varians)")
    return df_pca, pca


def run_umap(df_final_scaled: pd.DataFrame, verbose: bool = True):
    log = print if verbose else (lambda *a, **k: None)
    import umap  # diimpor di sini agar dependensi berat hanya dimuat saat dibutuhkan
    s = min(cfg.UMAP_SAMPLE, len(df_final_scaled))
    idx = np.random.RandomState(cfg.RANDOM_SEED).choice(len(df_final_scaled), s, replace=False)
    reducer = umap.UMAP(n_components=2, n_neighbors=cfg.UMAP_NEIGHBORS,
                        min_dist=cfg.UMAP_MIN_DIST, densmap=cfg.UMAP_DENSMAP,
                        random_state=cfg.RANDOM_SEED)
    emb = reducer.fit_transform(df_final_scaled.iloc[idx])
    df_umap = pd.DataFrame(emb, columns=["UMAP1", "UMAP2"],
                           index=df_final_scaled.index[idx])
    log(f"[reduce] UMAP: sample {s:,} -> 2D")
    return df_umap


def reduce(df_final_scaled: pd.DataFrame, verbose: bool = True):
    df_pca, pca = run_pca(df_final_scaled, verbose)
    try:
        df_umap = run_umap(df_final_scaled, verbose)
    except ImportError:
        print("[reduce] umap-learn belum terpasang; UMAP dilewati (jalankan: pip install umap-learn)")
        df_umap = pd.DataFrame()
    return df_pca, df_umap, pca
