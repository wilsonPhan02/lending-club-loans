"""Modul 00 — Pemuatan data & optimasi memori."""
import numpy as np
import pandas as pd

from . import config as cfg


def optimize_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    """Downcast kolom numerik (float64->float32, int64->downcast) untuk menekan memori."""
    for col in data.select_dtypes(include=["float64"]).columns:
        data[col] = pd.to_numeric(data[col], downcast="float")
    for col in data.select_dtypes(include=["int64"]).columns:
        data[col] = pd.to_numeric(data[col], downcast="integer")
    return data


def load_data(nrows: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    Muat dataset accepted. `nrows` (opsional) untuk uji cepat; jika None mengikuti config
    (USE_FULL / SAMPLE_SIZE).
    """
    if nrows is not None:
        df = pd.read_csv(cfg.DATA_PATH, low_memory=False, nrows=nrows)
    elif cfg.USE_FULL:
        df = pd.read_csv(cfg.DATA_PATH, low_memory=False)
    else:
        rng = np.random.RandomState(cfg.RANDOM_SEED)
        skip = sorted(rng.choice(np.arange(1, cfg.TOTAL_ROWS + 1),
                                 size=cfg.TOTAL_ROWS - cfg.SAMPLE_SIZE, replace=False))
        df = pd.read_csv(cfg.DATA_PATH, low_memory=False, skiprows=skip)

    df = optimize_dtypes(df)

    # Konversi kolom persen (tersimpan sebagai string '13.56%') -> numerik
    for c in ["int_rate", "revol_util"]:
        if c in df.columns and df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c].astype(str).str.replace("%", "", regex=False),
                                  errors="coerce")
    if verbose:
        mem = df.memory_usage(deep=True).sum() / 1e9
        print(f"[load] {df.shape[0]:,} baris x {df.shape[1]} kolom | {mem:.2f} GB")
    return df
