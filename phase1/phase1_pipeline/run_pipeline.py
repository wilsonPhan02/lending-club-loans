"""
Orkestrator pipeline Phase 1: load -> clean -> transform -> feature selection ->
dim. reduction -> simpan output.

Pemakaian:
    # dari root repo (folder yang memuat accepted_2007_to_2018Q4.csv)
    python -m phase1_pipeline.run_pipeline

    # atau dari notebook / skrip:
    from phase1_pipeline import run
    art = run()            # dict berisi seluruh artefak (df_final, df_scaled, df_pca, ...)
"""
import os

from . import config as cfg
from .data_loader import load_data
from .cleaning import clean
from .transform import transform
from .feature_selection import select_features
from .dim_reduction import reduce


def save_outputs(art: dict, verbose: bool = True) -> None:
    o = cfg.OUTPUTS
    to_write = {
        o["cleaned"]:     art["df_final"],
        o["no_winsor"]:   art["df_prewinsor"][art["final_features"]],
        o["scaled"]:      art["df_final_scaled"],
        o["pca"]:         art["df_pca"],
        o["arm"]:         art["arm_df"],
        o["loan_status"]: art["loan_status_ref"].to_frame("loan_status"),
        o["umap"]:        art["df_umap"],
    }
    for name, obj in to_write.items():
        obj.to_csv(name, index=False)
        if verbose:
            rel = os.path.relpath(name, cfg._ROOT).replace(os.sep, "/")
            print(f"  tersimpan: {rel:42s} {obj.shape}")


def run(nrows: int | None = None, save: bool = True, verbose: bool = True) -> dict:
    """Jalankan seluruh pipeline. Kembalikan dict artefak. `nrows` untuk uji cepat."""
    df = load_data(nrows=nrows, verbose=verbose)
    df, loan_status_ref = clean(df, verbose=verbose)
    df, df_scaled, arm_df, df_prewinsor = transform(df, verbose=verbose)
    df, df_scaled, df_final, df_final_scaled, final_features, mi_df = \
        select_features(df, df_scaled, verbose=verbose)
    df_pca, df_umap, pca = reduce(df_final_scaled, verbose=verbose)

    art = dict(df_final=df_final, df_final_scaled=df_final_scaled, df_prewinsor=df_prewinsor,
               df_pca=df_pca, df_umap=df_umap, arm_df=arm_df, mi_df=mi_df, pca=pca,
               final_features=final_features, loan_status_ref=loan_status_ref)
    if save:
        if verbose:
            print("Menyimpan output Phase 1 ...")
        save_outputs(art, verbose=verbose)
    if verbose:
        print(f"Pipeline selesai. Subset final: {len(final_features)} fitur; "
              f"PCA: {df_pca.shape[1]} komponen.")
    return art


if __name__ == "__main__":
    run()
