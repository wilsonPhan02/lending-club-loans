"""
Modul 01 — Data Cleaning.
Fix inkonsistensi tipe -> drop identifier/leakage -> tangani nulls (structural/MAR/MCAR)
-> hapus duplikat. Mengembalikan (df_fitur, loan_status_ref).
"""
import pandas as pd

from . import config as cfg


def clean(df: pd.DataFrame, verbose: bool = True):
    log = print if verbose else (lambda *a, **k: None)

    # --- 1. Perbaiki inkonsistensi tipe/format ---
    if "term" in df.columns and df["term"].dtype == "object":
        df["term"] = pd.to_numeric(
            df["term"].astype(str).str.replace("months", "", regex=False).str.strip(),
            errors="coerce").astype("Int64")
    if "emp_length" in df.columns and df["emp_length"].dtype == "object":
        df["emp_length"] = df["emp_length"].map(cfg.EMP_LENGTH_MAP)
    for c in ["int_rate", "revol_util"]:
        if c in df.columns and df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c].astype(str).str.replace("%", "", regex=False),
                                  errors="coerce")

    # --- 2. Drop identifier, teks bebas, kolom konstan ---
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    drop_now = [c for c in set(cfg.ID_TEXT_COLS + const_cols) if c in df.columns]
    df.drop(columns=drop_now, inplace=True, errors="ignore")
    log(f"[clean] drop identifier/teks/konstan: {len(drop_now)}")

    # --- 3. Drop post-loan (leakage) + tanggal ---
    post_loan = list(cfg.POST_LOAN_COLS) + [
        c for c in df.columns if c.startswith(cfg.POST_LOAN_PREFIXES)]
    drop_leak = [c for c in set(post_loan + cfg.DATE_COLS)
                 if c in df.columns and c != "loan_status"]
    df.drop(columns=drop_leak, inplace=True, errors="ignore")
    log(f"[clean] drop post-loan/leakage+tanggal: {len(drop_leak)}")

    # --- 4. Structural missing mths_since_* -> flag + sentinel ---
    mths_cols = [c for c in df.columns if c.startswith("mths_since")]
    for c in mths_cols:
        df[c + "_present"] = df[c].notna().astype("int8")
        df[c] = df[c].fillna(cfg.MTHS_SENTINEL)
    log(f"[clean] mths_since_* -> flag+sentinel: {len(mths_cols)}")

    # --- 5. Joint/sec_app -> drop sparse + flag is_joint ---
    is_joint = None
    if "application_type" in df.columns:
        is_joint = df["application_type"].astype(str).str.contains("Joint").astype("int8")
    joint_cols = [c for c in df.columns
                  if "joint" in c.lower() or c.lower().startswith("sec_app_")]
    df.drop(columns=joint_cols, inplace=True, errors="ignore")
    if is_joint is not None:
        df["is_joint"] = is_joint
        df.drop(columns=["application_type"], inplace=True, errors="ignore")

    # --- 6. Drop kolom high-missing (MAR temporal) di atas ambang ---
    miss = df.isnull().mean() * 100
    high_missing = [c for c in miss[miss > cfg.MISS_THRESHOLD].index if c != "loan_status"]
    df.drop(columns=high_missing, inplace=True, errors="ignore")
    log(f"[clean] drop missing >{cfg.MISS_THRESHOLD}%: {len(high_missing)}")

    # --- 7. Imputasi sisa missing ---
    if {"mort_acc", "total_acc"}.issubset(df.columns) and df["mort_acc"].isnull().any():
        grp_mean = df.groupby("total_acc")["mort_acc"].transform("mean")
        df["mort_acc"] = df["mort_acc"].fillna(grp_mean).fillna(df["mort_acc"].median())
    rem = df.isnull().sum(); rem = rem[rem > 0]
    num_na = [c for c in rem.index if pd.api.types.is_numeric_dtype(df[c]) and c != "loan_status"]
    cat_na = [c for c in rem.index if c not in num_na and c != "loan_status"]
    for c in num_na:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_na:
        df[c] = df[c].fillna(df[c].mode().iloc[0])
    if "loan_status" in df.columns and df["loan_status"].isnull().any():
        df = df[df["loan_status"].notna()].reset_index(drop=True)

    # --- 8. Hapus duplikat & pisahkan label referensi ---
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
    loan_status_ref = None
    if "loan_status" in df.columns:
        loan_status_ref = df["loan_status"].copy()
        df.drop(columns=["loan_status"], inplace=True)

    log(f"[clean] selesai: {df.shape[0]:,} baris x {df.shape[1]} fitur | "
        f"missing={int(df.isnull().sum().sum())}")
    return df, loan_status_ref
