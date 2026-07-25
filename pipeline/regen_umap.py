"""
Regenerasi `lending_club_umap.csv` dengan densMAP — tanpa menjalankan pipeline penuh.
Memakai `scaled_lending_club.csv` (= df_final_scaled) dan parameter dari config,
sehingga sampling & baris tetap selaras dengan Phase 2 (DBSCAN).

Pakai:  python regen_umap.py
"""
import numpy as np
import pandas as pd
import umap

from phase1_pipeline import config as cfg

X = pd.read_csv(cfg.OUTPUTS["scaled"])                       # subset final, scaled
rng = np.random.RandomState(cfg.RANDOM_SEED)
idx = rng.choice(len(X), min(cfg.UMAP_SAMPLE, len(X)), replace=False)

print(f"Menjalankan UMAP (densmap={cfg.UMAP_DENSMAP}, n_neighbors={cfg.UMAP_NEIGHBORS}) "
      f"pada {len(idx):,} baris ...")
reducer = umap.UMAP(n_components=2, n_neighbors=cfg.UMAP_NEIGHBORS,
                    min_dist=cfg.UMAP_MIN_DIST, densmap=cfg.UMAP_DENSMAP,
                    random_state=cfg.RANDOM_SEED)
emb = reducer.fit_transform(X.values[idx])
pd.DataFrame(emb, columns=["UMAP1", "UMAP2"]).to_csv(cfg.OUTPUTS["umap"], index=False)
print(f"Selesai. Tersimpan: {cfg.OUTPUTS['umap']}  {emb.shape}")
