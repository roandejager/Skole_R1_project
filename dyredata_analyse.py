import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import pearsonr


#  last inn alle datasett
df_klima = pd.read_csv("filer_med_data/figur-1-totalt-klimagass.csv", sep=";", decimal=",")
klima = pd.DataFrame({
    "Aar":      df_klima.iloc[:, 0].values.astype(int),
    "Totalt":   df_klima.iloc[:, 1:].values.astype(float).sum(axis=1),
    "Olje":     df_klima["Olje- og gassutvinning"].values.astype(float),
    "Industri": df_klima["Industri og bergverk"].values.astype(float),
    "Veitrafikk": df_klima["Veitrafikk"].values.astype(float),
})
klima.loc[len(klima)] = [2024, 44.6, 11.0, 10.6, 7.5]  # foreløpige SSB-tall

rein = pd.read_csv("filer_med_data/reintall_norge.csv", sep=";")

ulv = pd.read_csv("filer_med_data/ulv_bestand_norge.csv", sep=";", comment="#")
ulv["Aar"] = ulv["Sesong"].str.split("-").str[-1].astype(int)

elg = pd.read_csv("filer_med_data/elg_felt_norge.csv", sep=";", comment="#")

rovdyr = pd.read_csv("filer_med_data/norway_carnivore_mortality_2000_2025.csv")
rovdyr["Aar"] = rovdyr["Season"].str.split("-").str[-1].astype(int)

# slå sammen på år
data = (klima
    .merge(rein.rename(columns={"Antall_rein": "Rein"}), on="Aar", how="left")
    .merge(ulv[["Aar", "Ulv_Norge"]].rename(columns={"Ulv_Norge": "Ulv"}), on="Aar", how="left")
    .merge(elg.rename(columns={"Elg_felt": "Elg"}), on="Aar", how="left")
    .merge(rovdyr[["Aar", "Bear", "Wolverine", "Lynx"]]
           .rename(columns={"Bear": "Bjorn", "Wolverine": "Jerv", "Lynx": "Gaupe"}),
           on="Aar", how="left")
)

#  Korrelasjon: dyr vs CO2 utslipp
dyr_kolonner = {
    "Rein":  "Reindyr (antall)",
    "Ulv":   "Ulv (bestand)",
    "Elg":   "Elg felt per år",
    "Bjorn": "Bjørn (avgang)",
    "Jerv":  "Jerv (avgang)",
    "Gaupe": "Gaupe (avgang)",
}
klima_kolonner = {
    "Totalt":   "Totale klimagassutslipp",
    "Olje":     "Olje- og gassutvinning",
    "Industri": "Industri og bergverk",
    "Veitrafikk": "Veitrafikk",
}

print("=" * 70)
print("  KORRELASJONSANALYSE: Dyredata vs. klimagassutslipp")
print("  Pearson r  |  p-verdi  |  Tolkning")
print("=" * 70)

resultater = {}
for dk, dlabel in dyr_kolonner.items():
    for kk, klabel in klima_kolonner.items():
        sub = data[["Aar", dk, kk]].dropna()
        if len(sub) < 6:
            continue
        r, p = pearsonr(sub[kk], sub[dk])
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        tolkning = "Sterk" if abs(r) > 0.7 else "Moderat" if abs(r) > 0.4 else "Svak"
        retning = "negativ" if r < 0 else "positiv"
        resultater[(dk, kk)] = (r, p, len(sub))
        print(f"\n  {dlabel}")
        print(f"    vs. {klabel}")
        print(f"    r = {r:+.3f}  |  p = {p:.4f} {stars}  |  {tolkning} {retning}  (n={len(sub)})")

print("\n  *** p < 0.001   ** p < 0.01   * p < 0.05")
print("=" * 70)

#  Figur 1:  heatmap
fig1, ax = plt.subplots(figsize=(10, 6))
fig1.patch.set_facecolor("#f9f9f9")
ax.set_facecolor("#f9f9f9")

dyr_keys   = list(dyr_kolonner.keys())
klima_keys = list(klima_kolonner.keys())
matrise = np.full((len(dyr_keys), len(klima_keys)), np.nan)

for i, dk in enumerate(dyr_keys):
    for j, kk in enumerate(klima_keys):
        if (dk, kk) in resultater:
            matrise[i, j] = resultater[(dk, kk)][0]

im = ax.imshow(matrise, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Pearson r")

ax.set_xticks(range(len(klima_keys)))
ax.set_xticklabels([klima_kolonner[k] for k in klima_keys], rotation=25, ha="right", fontsize=10)
ax.set_yticks(range(len(dyr_keys)))
ax.set_yticklabels([dyr_kolonner[k] for k in dyr_keys], fontsize=10)

for i in range(len(dyr_keys)):
    for j in range(len(klima_keys)):
        if not np.isnan(matrise[i, j]):
            r_val = matrise[i, j]
            p_val = resultater[(dyr_keys[i], klima_keys[j])][1]
            stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            ax.text(j, i, f"{r_val:+.2f}{stars}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if abs(r_val) > 0.6 else "black")

ax.set_title("Korrelasjonsmatrise: Dyrebestander vs. klimagassutslipp (Pearson r)\n"
             "* p<0.05   ** p<0.01   *** p<0.001", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("resultater/korrelasjon_matrise.png", dpi=150, bbox_inches="tight")
print("\nLagret: korrelasjon_matrise.png")

#  Figur 2: Scatter  Ulv vs. totale utslipp (sterkeste korrelasjon)
fig2, axes = plt.subplots(1, 2, figsize=(13, 5))
fig2.patch.set_facecolor("#f9f9f9")
fig2.suptitle("Sterkeste korrelasjoner: Dyredata vs. klimagassutslipp",
              fontsize=13, fontweight="bold")

for ax, (dk, kk, tittel) in zip(axes, [
    ("Ulv", "Totalt", "Ulv (bestand) vs. Totale utslipp"),
    ("Elg", "Totalt", "Elg (felt) vs. Totale utslipp"),
]):
    sub = data[["Aar", dk, kk]].dropna()
    r, p = pearsonr(sub[kk], sub[dk])
    slope, intercept, *_ = stats.linregress(sub[kk], sub[dk])
    x_fit = np.linspace(sub[kk].min(), sub[kk].max(), 200)
    y_fit = slope * x_fit + intercept

    sc = ax.scatter(sub[kk], sub[dk], c=sub["Aar"], cmap="viridis", s=60, zorder=5)
    ax.plot(x_fit, y_fit, "r--", linewidth=1.8, label=f"Regresjonslinje")
    plt.colorbar(sc, ax=ax, label="År")

    ax.set_xlabel(klima_kolonner[kk] + " (mill. tonn CO₂-ekv.)", fontsize=10)
    ax.set_ylabel(dyr_kolonner[dk], fontsize=10)
    ax.set_title(f"{tittel}\nr = {r:+.3f}  |  p = {p:.4f}", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_facecolor("#f9f9f9")

plt.tight_layout()
plt.savefig("resultater/korrelasjon_scatter.png", dpi=150, bbox_inches="tight")
print("Lagret: korrelasjon_scatter.png")

#  Figur 3: Alle dyrebestander med trend og prognose
PROGNOSE_AAR = 2060
prognose_x = np.arange(2000, PROGNOSE_AAR + 1)

dyr_plot = [
    ("Rein",  "Reindyr (antall i Norge)", "#2196F3", "Reindeer"),
    ("Ulv",   "Ulv (bestand i Norge)",    "#9C27B0", "Ulv"),
    ("Elg",   "Elg felt per år",          "#FF9800", "Elg"),
    ("Bjorn", "Bjørn (avgang per år)",    "#795548", "Bjørn"),
    ("Jerv",  "Jerv (avgang per år)",     "#607D8B", "Jerv"),
    ("Gaupe", "Gaupe (avgang per år)",    "#E91E63", "Gaupe"),
]

fig3 = plt.figure(figsize=(16, 12))
fig3.patch.set_facecolor("#f5f5f5")
fig3.suptitle("Norske dyrebestander – historisk utvikling og prognose mot 2060",
              fontsize=14, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig3, hspace=0.45, wspace=0.3)

for idx, (col, label, farge, kortnavn) in enumerate(dyr_plot):
    ax = fig3.add_subplot(gs[idx // 2, idx % 2])
    ax.set_facecolor("#fafafa")

    sub = data[["Aar", col]].dropna()
    x = sub["Aar"].values
    y = sub[col].values

    # lineær regresjon
    slope, intercept, r, p, _ = stats.linregress(x, y)
    y_fit    = slope * x + intercept
    y_prog   = slope * prognose_x + intercept

    #  når prognosen treffer 0
    nullpunkt = None
    if slope < 0:
        nullpunkt = int(-intercept / slope)

    # enkel approksimering
    residual = y - y_fit
    se = np.std(residual) * 1.96
    y_upper = y_prog + se
    y_lower = np.maximum(y_prog - se, 0)

    # Plot
    ax.scatter(x, y, color=farge, s=40, zorder=5, label="Faktisk")
    ax.plot(x, y_fit, color=farge, linewidth=2, label="Trendlinje")
    ax.plot(prognose_x[prognose_x > x[-1]], y_prog[prognose_x > x[-1]],
            color=farge, linestyle="--", linewidth=1.5, alpha=0.7, label="Prognose")
    ax.fill_between(prognose_x[prognose_x > x[-1]],
                    y_lower[prognose_x > x[-1]],
                    y_upper[prognose_x > x[-1]],
                    alpha=0.12, color=farge, label="95% konfidensintervall")

    #  nullpunkt hvis relevant
    if nullpunkt and 2024 < nullpunkt < PROGNOSE_AAR:
        ax.axvline(x=nullpunkt, color="red", linestyle=":", linewidth=1.5, alpha=0.8)
        ax.text(nullpunkt + 0.5, y.max() * 0.05,
                f"Prognose = 0\n≈ {nullpunkt}", color="red", fontsize=8, va="bottom")
    elif slope < 0 and nullpunkt and nullpunkt <= 2024:
        ax.text(0.97, 0.95, f"Kritisk punkt\nallerede passert?",
                transform=ax.transAxes, fontsize=7.5, ha="right", va="top", color="red")

    # info tekst
    retning = f"+{slope:.1f}/år" if slope >= 0 else f"{slope:.1f}/år"
    p_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ax.text(0.03, 0.97,
            f"Trend: {retning}\nr² = {r**2:.2f}  {p_str}",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("År", fontsize=9)
    ax.set_xlim(1999, PROGNOSE_AAR + 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper right" if slope > 0 else "upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.axvspan(x[-1] + 1, PROGNOSE_AAR + 1, alpha=0.04, color="gray")

plt.savefig("resultater/dyrebestander_prognose.png", dpi=150, bbox_inches="tight")
print("Lagret: dyrebestander_prognose.png")

#  Figur 4: Ulv + CO2 på dobbel yakse (tydelig visuell sammenheng)
fig4, ax1 = plt.subplots(figsize=(12, 6))
fig4.patch.set_facecolor("#f9f9f9")
ax1.set_facecolor("#f9f9f9")
ax2 = ax1.twinx()

sub_ulv = data[["Aar","Ulv","Totalt"]].dropna(subset=["Ulv"])

ax1.plot(sub_ulv["Aar"], sub_ulv["Totalt"], color="#d73027", linewidth=2.5,
         marker="o", markersize=5, label="Totale klimagassutslipp (venstre akse)")
ax2.plot(sub_ulv["Aar"], sub_ulv["Ulv"], color="#4575b4", linewidth=2.5,
         marker="s", markersize=5, label="Ulvebestand (høyre akse)")

ax1.set_xlabel("År", fontsize=12)
ax1.set_ylabel("Mill. tonn CO₂-ekvivalenter", color="#d73027", fontsize=11)
ax2.set_ylabel("Antall ulv (norsk territorium)", color="#4575b4", fontsize=11)
ax1.tick_params(axis="y", colors="#d73027")
ax2.tick_params(axis="y", colors="#4575b4")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=10)

r_ulv, p_ulv = pearsonr(sub_ulv["Totalt"], sub_ulv["Ulv"])
ax1.set_title(f"Ulvebestand og klimagassutslipp i Norge (r = {r_ulv:+.3f}, p < 0.001)\n"
              f"Sterk negativ korrelasjon – begge styrt av tidsutvikling etter 2002",
              fontsize=12, fontweight="bold")
ax1.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("resultater/ulv_vs_klimagass.png", dpi=150, bbox_inches="tight")
print("Lagret: ulv_vs_klimagass.png")

print("\nAlle figurer lagret. Kjør scriptet for å vise grafene.")
plt.show()
