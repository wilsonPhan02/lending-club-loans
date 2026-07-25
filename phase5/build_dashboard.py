"""
Dashboard Phase 5 (phase5/dashboard.html).
Membaca output fase dari data/processed/ lalu merakit dashboard Plotly
satu file (tanpa server): cluster map, radar profil, rule network,
rule scatter, outlier plot, validasi bad-rate, dan distribusi populasi.

Jalankan:  python phase5/build_dashboard.py
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda f: os.path.join(ROOT, "data", "processed", f)
RANDOM_SEED = 42

# ----------------------------------------------------------------- load
profile  = pd.read_csv(P("cleaned_lending_club_no_winsorization.csv"))
kmeans   = pd.read_csv(P("phase2_kmeans_labels.csv"))["kmeans_cluster"].values
prof_tbl = pd.read_csv(P("phase2_cluster_profiles.csv"))
umap_df  = pd.read_csv(P("lending_club_umap.csv"))
dbs      = pd.read_csv(P("phase2_dbscan_labels.csv"))
rules    = pd.read_csv(P("phase3_association_rules.csv"))
anom     = pd.read_csv(P("phase4_anomalies.csv"))
loan     = pd.read_csv(P("loan_status_reference.csv"))["loan_status"]

# Reproduksi indeks sample UMAP (identik dengan pipeline Phase 1: RandomState(42).choice)
samp_idx = np.random.RandomState(RANDOM_SEED).choice(len(profile), len(umap_df), replace=False)
km_u    = kmeans[samp_idx]
noise_u = np.zeros(len(umap_df), bool); noise_u[dbs["is_noise"].values == 1] = True
co_u    = loan.iloc[samp_idx].eq("Charged Off").values
names   = dict(zip(prof_tbl["cluster"], prof_tbl["profil"]))
print(f"profile {profile.shape} | umap {umap_df.shape} | rules {len(rules):,} | anomali {len(anom):,}")

kpis = [
    (f"{len(profile):,}".replace(",", "."), "Total pinjaman"),
    (str(profile.shape[1]),                 "Fitur final"),
    (str(prof_tbl["cluster"].nunique()),    "Cluster (K-Means)"),
    (f"{len(rules):,}".replace(",", "."),   "Rule asosiasi (retained)"),
    (f"{len(anom):,}".replace(",", "."),    "Anomali terkonfirmasi"),
    (f"{(anom['classification']=='data_error').sum():,}".replace(",", "."), "Data error"),
]

# ----------------------------------------------------------------- charts
LAY = dict(font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#0F172A"),
           paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
           margin=dict(l=50, r=20, t=52, b=45))
CLUST_COLORS = ["#E4572E", "#1B998B", "#4C6EF5"]

def fig_clustermap():
    # Render 40k dari 100k titik UMAP (kepadatan visual identik, ukuran file 1/3);
    # statistik/kesimpulan tidak bergantung pada subsample render ini.
    N_MAP = 40_000
    ridx = np.random.RandomState(RANDOM_SEED).choice(len(umap_df), N_MAP, replace=False)
    x = umap_df["UMAP1"].values[ridx].round(2); y = umap_df["UMAP2"].values[ridx].round(2)
    km_m, noise_m, co_m = km_u[ridx], noise_u[ridx], co_u[ridx]
    fig = go.Figure()
    for i, c in enumerate(sorted(set(km_m))):
        m = km_m == c
        fig.add_scattergl(x=x[m], y=y[m], mode="markers", name=f"{names.get(c, c)} (C{c})",
                          marker=dict(size=2.4, color=CLUST_COLORS[i], opacity=0.45), visible=True)
    fig.add_scattergl(x=x[~noise_m], y=y[~noise_m], mode="markers", name="core (DBSCAN)",
                      marker=dict(size=2.4, color="#94A3B8", opacity=0.35), visible=False)
    fig.add_scattergl(x=x[noise_m], y=y[noise_m], mode="markers", name="noise / outlier",
                      marker=dict(size=5, color="#0B2545", opacity=0.85), visible=False)
    fig.add_scattergl(x=x[~co_m], y=y[~co_m], mode="markers", name="lancar/lainnya",
                      marker=dict(size=2.4, color="#CBD5E1", opacity=0.35), visible=False)
    fig.add_scattergl(x=x[co_m], y=y[co_m], mode="markers", name="Charged Off",
                      marker=dict(size=2.8, color="#E4572E", opacity=0.5), visible=False)
    nk = len(set(km_m))
    vis = lambda a, b: [a <= j < b for j in range(nk + 4)]
    fig.update_layout(LAY, title="Cluster Map — UMAP 2D (render 40k dari sampel 100k)",
        xaxis_title="UMAP-1", yaxis_title="UMAP-2", legend=dict(itemsizing="constant"),
        updatemenus=[dict(x=1.0, y=1.14, xanchor="right", type="dropdown", buttons=[
            dict(label="Warna: K-Means",      method="update", args=[{"visible": vis(0, nk)}]),
            dict(label="Warna: DBSCAN noise", method="update", args=[{"visible": vis(nk, nk+2)}]),
            dict(label="Warna: Charged-off",  method="update", args=[{"visible": vis(nk+2, nk+4)}]),
        ])])
    return fig

def fig_radar():
    feat = ["annual_inc","fico_range_low","revol_util","dti","loan_amnt","term","emp_length","grade"]
    means = profile.assign(cluster=kmeans).groupby("cluster")[feat].mean()
    z = (means - profile[feat].mean()) / profile[feat].std()
    fig = go.Figure()
    for i, c in enumerate(z.index):
        v = z.loc[c].tolist(); v += v[:1]
        fig.add_scatterpolar(r=v, theta=feat + feat[:1], name=f"{names.get(c, c)} (C{c})",
                             fill="toself", line=dict(color=CLUST_COLORS[i]))
    fig.update_layout(LAY, title="Sidik-jari Profil Cluster (z-score vs populasi)")
    return fig

def fig_sizes():
    lbl = [f"{names.get(c, c)}" for c in prof_tbl["cluster"]]
    fig = go.Figure()
    fig.add_bar(x=lbl, y=prof_tbl["size_%"], name="Ukuran (%)", marker_color="#1B998B",
                text=[f"{v}%" for v in prof_tbl["size_%"]], textposition="outside")
    fig.add_bar(x=lbl, y=prof_tbl["CO_%"], name="Charged-off (%)", marker_color="#E4572E",
                text=[f"{v}%" for v in prof_tbl["CO_%"]], textposition="outside")
    fig.update_layout(LAY, title="Risiko & Ukuran per Cluster", barmode="group", yaxis_title="%")
    return fig

def fig_rulenet(top_n=12):
    top = rules.nlargest(top_n, "lift")
    items = sorted(set(top["antecedents"].str.split(", ").sum() + top["consequents"].str.split(", ").sum()))
    ang = np.linspace(0, 2*np.pi, len(items), endpoint=False)
    pos = {it: (np.cos(a), np.sin(a)) for it, a in zip(items, ang)}
    fig = go.Figure()
    lmin, lmax = top["lift"].min(), top["lift"].max()
    for _, r in top.iterrows():
        for a in r["antecedents"].split(", "):
            for c in r["consequents"].split(", "):
                w = 1 + 4*(r["lift"]-lmin)/max(lmax-lmin, 1e-9)
                fig.add_scatter(x=[pos[a][0], pos[c][0]], y=[pos[a][1], pos[c][1]], mode="lines",
                                line=dict(width=w, color="rgba(27,153,139,0.45)"),
                                hoverinfo="text", showlegend=False,
                                text=f"{r['antecedents']} → {r['consequents']}<br>lift={r['lift']:.2f} conf={r['confidence']:.2f}")
    short = [i.split("_", 1)[-1].replace("_", " ") for i in items]
    fig.add_scatter(x=[pos[i][0] for i in items], y=[pos[i][1] for i in items],
                    mode="markers+text", text=short, textposition="top center",
                    textfont=dict(size=9), marker=dict(size=11, color="#0B2545"), showlegend=False)
    fig.update_layout(LAY, title=f"Jaringan Aturan Asosiasi ({top_n} rule lift-tertinggi)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
    return fig

def fig_rulescatter():
    steps_v = [1.15, 1.5, 2.0, 2.5, 3.0, 3.5]
    fig = go.Figure()
    for i, t in enumerate(steps_v):
        m = rules["lift"] >= t
        fig.add_scattergl(x=rules.loc[m,"support"], y=rules.loc[m,"confidence"], mode="markers",
            visible=(i == 0), showlegend=False,
            marker=dict(size=6, color=rules.loc[m,"lift"], colorscale="Viridis",
                        colorbar=dict(title="Lift"), opacity=0.65),
            text=[f"{a} → {c}<br>lift={l:.2f}" for a, c, l in
                  zip(rules.loc[m,"antecedents"], rules.loc[m,"consequents"], rules.loc[m,"lift"])],
            hoverinfo="text")
    fig.update_layout(LAY, title=f"Semua {len(rules):,} Rule — Support vs Confidence (warna = Lift)".replace(",", "."),
        xaxis_title="Support", yaxis_title="Confidence",
        sliders=[dict(active=0, currentvalue=dict(prefix="min lift ≥ "), pad=dict(t=28),
                      steps=[dict(label=f"{t}", method="update",
                                  args=[{"visible": [j == i for j in range(len(steps_v))]}])
                             for i, t in enumerate(steps_v)])])
    return fig

def fig_outlier():
    s = anom.sample(min(20_000, len(anom)), random_state=RANDOM_SEED)
    colors = {"rare_case":"#1B998B", "risk_signal":"#E4572E", "data_error":"#0B2545"}
    fig = go.Figure()
    for k, g in s.groupby("classification"):
        fig.add_scattergl(x=g["annual_inc"].clip(upper=5e6), y=g["dti"].clip(upper=120),
            mode="markers", name=f"{k} (n={(anom['classification']==k).sum():,})".replace(",", "."),
            marker=dict(size=4, color=colors[k], opacity=0.5))
    fig.update_layout(LAY, title="Outlier Plot — anomali di ruang fitur (sampel 20k, di-clip)",
                      xaxis_title="Annual income (clip $5jt)", yaxis_title="DTI (clip 120)")
    return fig

def fig_riskvalid():
    co_status = ["Charged Off", "Default", "Late (31-120 days)", "Late (16-30 days)",
                 "Does not meet the credit policy. Status:Charged Off"]
    pop = loan.isin(co_status).mean()*100
    anm = anom["loan_status"].isin(co_status).mean()*100
    rsk = anom.loc[anom["classification"]=="risk_signal", "loan_status"].isin(co_status).mean()*100
    fig = go.Figure(go.Bar(x=["Populasi", "Semua anomali", "risk_signal"], y=[pop, anm, rsk],
        marker_color=["#94A3B8", "#F4A259", "#E4572E"],
        text=[f"{v:.1f}%" for v in [pop, anm, rsk]], textposition="outside"))
    fig.update_layout(LAY, title="Validasi Risiko — bad-rate (%; Charged Off+Default+Late)",
                      yaxis_title="bad-rate %")
    return fig

def fig_grade():
    g = profile["grade"].map({1:"A",2:"B",3:"C",4:"D",5:"E",6:"F",7:"G"}).value_counts().sort_index()
    fig = go.Figure(go.Bar(x=g.index, y=g.values, marker_color="#1B998B"))
    fig.update_layout(LAY, title="Distribusi Grade (populasi)", xaxis_title="Grade", yaxis_title="Jumlah")
    return fig

def fig_loanstatus():
    ls = loan.value_counts().head(7)
    fig = go.Figure(go.Bar(y=ls.index[::-1], x=ls.values[::-1], orientation="h", marker_color="#4C6EF5"))
    fig.update_layout(LAY, title="Distribusi Loan Status (7 teratas)", xaxis_title="Jumlah",
                      margin=dict(l=210, r=20, t=52, b=45))
    return fig

figs = {
    "clustermap":  fig_clustermap(), "profileradar": fig_radar(),  "profilebar": fig_sizes(),
    "rulenet":     fig_rulenet(),    "rulescatter": fig_rulescatter(),
    "outlierplot": fig_outlier(),    "riskvalid":   fig_riskvalid(),
    "gradedist":   fig_grade(),      "loandist":    fig_loanstatus(),
}
print("chart siap:", list(figs))

# ----------------------------------------------------------------- assemble
CSS = """body{margin:0;background:#F0F4F8;color:#0F172A;font-family:'Inter','Segoe UI',system-ui,sans-serif}
header{background:linear-gradient(135deg,#071a36 0%,#0B2545 40%,#1a3a6e 100%);color:#fff;padding:30px 40px 26px}
h1{margin:0;font-size:25px;font-weight:800;letter-spacing:-0.4px}
.lead{color:#94A3B8;margin:8px 0 0;font-size:13.5px;max-width:820px;line-height:1.55}
.kpis{display:flex;flex-wrap:wrap;gap:14px;padding:22px 40px 4px}
.kpi{background:#fff;border-radius:12px;border-top:3px solid #1B998B;box-shadow:0 1px 3px rgba(11,37,69,.10);padding:15px 20px;min-width:150px;flex:1}
.kv{font-size:25px;font-weight:800;color:#0B2545}.kl{color:#475569;font-size:11px;margin-top:3px;text-transform:uppercase;letter-spacing:.5px}
section{padding:14px 40px 6px}
h2{font-size:19px;font-weight:800;color:#0B2545;margin:22px 0 2px}
.sub{color:#475569;font-size:13px;margin:0 0 12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:16px}
.card{background:#fff;border-radius:12px;box-shadow:0 1px 3px rgba(11,37,69,.10);padding:8px}
footer{color:#64748B;font-size:12px;padding:18px 40px 26px}"""

def frag(name):
    return figs[name].to_html(full_html=False, include_plotlyjs=False,
                              div_id=name, config={"displaylogo": False})

sections = [
    ("1 · Segmentasi Peminjam (Phase 2)",
     "K-Means (K=2) pada ruang PCA; profil divalidasi Hierarchical (ARI 0,357) dan DBSCAN. "
     "Gunakan dropdown pada cluster map untuk mengganti pewarnaan.",
     ["clustermap", "profileradar", "profilebar"]),
    ("2 · Association Rule Mining (Phase 3)",
     f"Apriori pada sample acak 250k; {len(rules):,} rule retained (lift>1,15; conf>0,5). "
     "Gunakan slider untuk memfilter minimum lift.".replace(",", ".", 1),
     ["rulenet", "rulescatter"]),
    ("3 · Anomaly Detection (Phase 4)",
     "Anomali terkonfirmasi ≥2 metode (IQR/Z-score/Mahalanobis/Isolation Forest), "
     "diklasifikasi data_error / risk_signal / rare_case.",
     ["outlierplot", "riskvalid"]),
    ("4 · Distribusi Populasi", "Konteks populasi untuk membaca temuan.", ["gradedist", "loandist"]),
]

html = ['<!DOCTYPE html><html lang="id"><head><meta charset="utf-8">',
        "<title>Lending Club — KDD Dashboard</title>",
        '<script src="https://cdn.plot.ly/plotly-3.7.0.min.js"></script>',
        f"<style>{CSS}</style></head><body>",
        "<header><h1>Lending Club — KDD Discovery Dashboard</h1>",
        '<p class="lead">Ringkasan visual Fase 2–4: segmentasi peminjam, aturan asosiasi, dan '
        "deteksi anomali. Seluruh chart dihasilkan dari output pipeline (bukan data sintetis).</p></header>",
        '<div class="kpis">']
for v, l in kpis:
    html.append(f'<div class="kpi"><div class="kv">{v}</div><div class="kl">{l}</div></div>')
html.append("</div>")
for title, sub, chart_ids in sections:
    html.append(f'<section><h2>{title}</h2><p class="sub">{sub}</p><div class="grid">')
    for cid in chart_ids:
        html.append(f'<div class="card">{frag(cid)}</div>')
    html.append("</div></section>")
html.append("<footer>Kelompok 2 · Data Mining · Lending Club Loans</footer></body></html>")

out = os.path.join(ROOT, "phase5", "dashboard.html")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(html))
print(f"Tersimpan: {out} ({os.path.getsize(out)/1e6:.1f} MB)")
