"""
Modul 03 — Feature Selection.
Korelasi (buang multikolinearitas, tie-break berbasis prioritas, target diproteksi) ->
Mutual Information terhadap `grade` (entropy) -> subset hybrid core+MI (anti-redundan).
Mengembalikan (df, df_scaled, df_final, df_final_scaled, final_features, mi_df).
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from . import config as cfg


def _correlation_prune(df, df_scaled, verbose):
    log = print if verbose else (lambda *a, **k: None)
    priority = cfg.CORR_PRIORITY

    def rank(c):
        return priority.index(c) if c in priority else len(priority)

    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = [(a, b, upper.loc[a, b]) for a in upper.index for b in upper.columns
             if pd.notna(upper.loc[a, b]) and upper.loc[a, b] > cfg.CORR_THRESHOLD]
    pairs.sort(key=lambda t: -t[2])

    to_drop = set()
    for a, b, _ in pairs:
        if a in to_drop or b in to_drop:
            continue
        loser = a if rank(a) > rank(b) else b
        if loser == cfg.MI_TARGET:                # proteksi mutlak target
            loser = b if a == cfg.MI_TARGET else a
        to_drop.add(loser)

    assert cfg.MI_TARGET not in to_drop, "target grade tidak boleh dibuang"
    df.drop(columns=list(to_drop), inplace=True, errors="ignore")
    df_scaled.drop(columns=[c for c in to_drop if c in df_scaled.columns],
                   inplace=True, errors="ignore")
    log(f"[select] korelasi |r|>{cfg.CORR_THRESHOLD}: buang {len(to_drop)} -> {df.shape[1]} fitur")
    return df, df_scaled


def _mutual_information(df, df_scaled, verbose):
    log = print if verbose else (lambda *a, **k: None)
    if cfg.MI_TARGET not in df_scaled.columns:
        raise RuntimeError("grade hilang sebelum MI — periksa korelasi-prune.")
    s = min(cfg.MI_SAMPLE, len(df_scaled))
    idx = np.random.RandomState(cfg.RANDOM_SEED).choice(len(df_scaled), s, replace=False)
    y = df.iloc[idx][cfg.MI_TARGET].astype(int)
    X = df_scaled.iloc[idx].drop(columns=[cfg.MI_TARGET])
    mi = mutual_info_classif(X, y, random_state=cfg.RANDOM_SEED, n_neighbors=cfg.MI_NEIGHBORS)
    mi_df = (pd.DataFrame({"feature": X.columns, "MI": mi})
             .sort_values("MI", ascending=False).reset_index(drop=True))
    log(f"[select] MI dihitung (sample {s:,}); top: "
        f"{mi_df.iloc[0]['feature']}={mi_df.iloc[0]['MI']:.3f}")
    return mi_df


def _build_subset(df, df_scaled, mi_df, verbose):
    log = print if verbose else (lambda *a, **k: None)
    core = [c for c in cfg.MINING_CORE if c in df_scaled.columns]
    final = list(core)
    corr_abs = df_scaled.corr().abs()
    for f in mi_df["feature"]:
        if len(final) >= cfg.TARGET_K:
            break
        if f in final:
            continue
        sel = [c for c in final if c in corr_abs.columns]
        if f in corr_abs.columns and corr_abs.loc[f, sel].max() > cfg.CORR_GUARD:
            continue                              # tolak redundan
        final.append(f)
    log(f"[select] subset final: {len(final)} fitur -> {final}")
    return df[final].copy(), df_scaled[final].copy(), final


def select_features(df, df_scaled, verbose: bool = True):
    df, df_scaled = _correlation_prune(df, df_scaled, verbose)
    mi_df = _mutual_information(df, df_scaled, verbose)
    df_final, df_final_scaled, final = _build_subset(df, df_scaled, mi_df, verbose)
    return df, df_scaled, df_final, df_final_scaled, final, mi_df
