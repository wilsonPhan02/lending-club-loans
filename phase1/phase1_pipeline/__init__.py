"""Pipeline preprocessing Phase 1 — Lending Club (KDD)."""
from . import config
from .data_loader import load_data
from .cleaning import clean
from .transform import transform
from .feature_selection import select_features
from .dim_reduction import reduce

# Catatan: `run` TIDAK di-import di sini agar `python -m phase1_pipeline.run_pipeline`
# tidak memicu double-import. Gunakan `from phase1_pipeline.run_pipeline import run`,
# atau jalankan `python -m phase1_pipeline`.

__all__ = ["config", "load_data", "clean", "transform",
           "select_features", "reduce"]
