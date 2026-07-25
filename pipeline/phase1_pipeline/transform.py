"""
Modul 02 — Data Transformation.
Encoding (ordinal+biner, nominal di-drop) -> binning (untuk Phase 3 ARM) ->
snapshot pre-winsor (untuk Phase 4) -> winsorization -> log kondisional ->
buang quasi-constant -> StandardScaler.
Mengembalikan (df, df_scaled, arm_df, df_prewinsor).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as cfg


def transform(df: pd.DataFrame, verbose: bool = True):
    log = print if verbose else (lambda *a, **k: None)

    # --- 1. Encoding ---
    nominal_for_arm = [c for c in cfg.NOMINAL_FOR_ARM if c in df.columns]
    df_nominal = df[nominal_for_arm].copy()

    if "term" in df.columns:
        df["term"] = df["term"].astype("float32")
    if "grade" in df.columns and df["grade"].dtype == "object":
        df["grade"] = df["grade"].map(cfg.GRADE_MAP).astype("float32")
    if "sub_grade" in df.columns and df["sub_grade"].dtype == "object":
        sg = sorted(df["sub_grade"].dropna().unique())
        df["sub_grade"] = df["sub_grade"].map({s: i + 1 for i, s in enumerate(sg)}).astype("float32")
    if "initial_list_status" in df.columns and df["initial_list_status"].dtype == "object":
        df["initial_list_status"] = (df["initial_list_status"] == "w").astype("int8")

    drop_nominal = [c for c in cfg.DROP_NOMINAL if c in df.columns]
    df.drop(columns=drop_nominal, inplace=True, errors="ignore")
    leftover_obj = df.select_dtypes(include="object").columns.tolist()  # catch-all nominal
    if leftover_obj:
        df.drop(columns=leftover_obj, inplace=True)
    log(f"[transform] encoding selesai; nominal di-drop: {len(drop_nominal) + len(leftover_obj)}")

    # --- 2. Binning (nilai asli) untuk reporting & Phase 3 ARM ---
    df_binned = pd.DataFrame(index=df.index)
    for col, (q, labels) in cfg.BIN_SPECS.items():
        if col in df.columns:
            df_binned[col + "_bin"] = pd.qcut(df[col], q=q, labels=labels, duplicates="drop")
    arm_df = pd.concat([df_binned, df_nominal.reset_index(drop=True)], axis=1)

    # --- 3. Snapshot nilai asli (pre-winsor, pre-log) untuk Phase 4 ---
    df_prewinsor = df.copy()

    # Fitur kontinu (kandidat winsor & log): numerik minus ordinal/biner/flag
    exclude_cont = set(cfg.LOG_EXCLUDE_BASE + [c for c in df.columns if c.endswith("_present")])
    cont = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cont]

    # --- 4. Winsorization DATA-DRIVEN (fitur heavy-tail |skew|>ambang; guard q99>median) ---
    if cfg.WINSORIZE:
        sk = df[cont].skew().abs()
        targets = [c for c in cont if sk[c] > cfg.WINSOR_SKEW_THRESHOLD]
        wins = []
        for c in targets:
            q50, q99 = df[c].quantile(0.50), df[c].quantile(cfg.WINSOR_HIGH)
            if q99 > q50:                      # guard: hindari kolaps zero-inflated
                df[c] = df[c].clip(df[c].quantile(cfg.WINSOR_LOW), q99)
                wins.append(c)
        log(f"[transform] winsorization 1/99 (|skew|>{cfg.WINSOR_SKEW_THRESHOLD}): {len(wins)} fitur")

    # --- 5. Log transform kondisional (hanya bila |skew| turun) ---
    skew = df[cont].skew()
    cand = [c for c in cont if abs(skew[c]) > cfg.LOG_SKEW_THRESHOLD and (df[c] >= 0).all()]
    applied = 0
    for c in cand:
        trans = np.log1p(df[c])
        if abs(trans.skew()) < abs(df[c].skew()):
            df[c] = trans
            applied += 1
    log(f"[transform] log1p: {applied}/{len(cand)} fitur (kondisional)")

    # --- 6. Buang fitur quasi-constant (nilai dominan >= ambang) ---
    dom = df.apply(lambda s: s.value_counts(normalize=True, dropna=False).iloc[0])
    quasi = [c for c in dom[dom >= cfg.QUASI_CONST_RATIO].index if c not in cfg.PROTECT_COLS]
    if quasi:
        df.drop(columns=quasi, inplace=True)
    log(f"[transform] drop quasi-constant: {len(quasi)}")

    # --- 7. Standard scaling ---
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns,
                             index=df.index).astype("float32")
    log(f"[transform] scaling selesai: {df_scaled.shape}")

    return df, df_scaled, arm_df, df_prewinsor
