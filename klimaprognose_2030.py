import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


df = pd.read_csv("figur-1-totalt-klimagass.csv", sep=";", decimal=",")

aar = df.iloc[:, 0].values.astype(int)
totalt = df.iloc[:, 1:].values.astype(float).sum(axis=1)

# Legg til 2024 fra SSB-oversikten (foreløpige tall)
aar = np.append(aar, 2024)
totalt = np.append(totalt, 44.6)


# Lineær regresjon på hele perioden
koeff_lin = np.polyfit(aar, totalt, 1)
modell_lin = np.poly1d(koeff_lin)

# Lineær regresjon fra 2005–2024 (nedgangstrend)
mask = aar >= 2005
koeff_ned = np.polyfit(aar[mask], totalt[mask], 1)
modell_ned = np.poly1d(koeff_ned)

# klimamål
prognose_aar = np.arange(1990, 2036)
mal_2030 = totalt[0] * 0.45   # 55 % reduksjon fra 1990 = 45 % igjen
baseline_1990 = totalt[0]

prognose_lin = modell_lin(prognose_aar)
prognose_ned = modell_ned(prognose_aar)

verdi_2030_lin = modell_lin(2030)
verdi_2030_ned = modell_ned(2030)
gap_lin = verdi_2030_lin - mal_2030
gap_ned = verdi_2030_ned - mal_2030

# koeffisienter
print("=" * 55)
print("  REGRESJONSKOEFFISIENTER")
print("=" * 55)
print(f"\nModell 1 (lineær, 1990–2024):")
print(f"  f(x) = {koeff_lin[0]:.4f}x + ({koeff_lin[1]:.2f})")
print(f"\nModell 2 (lineær, nedgang 2005–2024):")
print(f"  g(x) = {koeff_ned[0]:.4f}x + ({koeff_ned[1]:.2f})")
print(f"\nNorsk klimamål 2030 (55 % under 1990-nivå):")
print(f"  Mål: {mal_2030:.2f} millioner tonn CO₂-ekv.")
print(f"\nPrognose 2030:")
print(f"  Modell 1: {verdi_2030_lin:.1f} mill. tonn  (gap: +{gap_lin:.1f})")
print(f"  Modell 2: {verdi_2030_ned:.1f} mill. tonn  (gap: +{gap_ned:.1f})")
print("=" * 55)

# plott
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("#f9f9f9")
ax.set_facecolor("#f9f9f9")

# faktiske utslipp
ax.scatter(aar, totalt, color="#2c7bb6", s=50, zorder=5, label="Faktiske utslipp (SSB)")

#regresjonslinjer
ax.plot(prognose_aar, prognose_lin, color="#d73027", linestyle="--", linewidth=1.8,
        label=f"Modell 1 – lineær (1990–2024): f(x) = {koeff_lin[0]:.3f}x + {koeff_lin[1]:.0f}")
ax.plot(prognose_aar, prognose_ned, color="#fc8d59", linestyle="-.", linewidth=1.8,
        label=f"Modell 2 – trend fra 2005: g(x) = {koeff_ned[0]:.3f}x + {koeff_ned[1]:.0f}")

# klimamål-linje
ax.axhline(y=mal_2030, color="#1a9641", linestyle=":", linewidth=2,
           label=f"Klimamål 2030: {mal_2030:.1f} mill. tonn (–55 % fra 1990)")

#  vertikale referanselinjer
ax.axvline(x=2030, color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax.axvline(x=2024, color="#2c7bb6", linestyle=":", linewidth=1, alpha=0.4)

# annotasjoner for prognose 2030
ax.annotate(f"Modell 1: {verdi_2030_lin:.1f}",
            xy=(2030, verdi_2030_lin), xytext=(2031, verdi_2030_lin + 1.5),
            fontsize=9, color="#d73027",
            arrowprops=dict(arrowstyle="->", color="#d73027", lw=1.2))

ax.annotate(f"Modell 2: {verdi_2030_ned:.1f}",
            xy=(2030, verdi_2030_ned), xytext=(2031, verdi_2030_ned - 3),
            fontsize=9, color="#fc8d59",
            arrowprops=dict(arrowstyle="->", color="#fc8d59", lw=1.2))

ax.annotate(f"Mål: {mal_2030:.1f}",
            xy=(2030, mal_2030), xytext=(2031, mal_2030 - 3),
            fontsize=9, color="#1a9641",
            arrowprops=dict(arrowstyle="->", color="#1a9641", lw=1.2))

# gappiler mellom prognose og mål
ax.annotate("", xy=(2030, mal_2030), xytext=(2030, verdi_2030_ned),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax.text(2030.4, (mal_2030 + verdi_2030_ned) / 2,
        f"Gap:\n{gap_ned:.1f} mill. tonn", fontsize=8.5, va="center")

# akse og tittel
ax.set_xlim(1988, 2036)
ax.set_ylim(0, 65)
ax.set_xlabel("År", fontsize=12)
ax.set_ylabel("Millioner tonn CO₂-ekvivalenter", fontsize=12)
ax.set_title("Norges klimagassutslipp 1990–2024 med prognose mot 2030\n"
             "og sammenligning mot klimamål (55 % reduksjon fra 1990)",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# usikkerhetsområde etter 2024
ax.axvspan(2025, 2035, alpha=0.06, color="gray", label="Prognoseperiode")
ax.text(2026, 5, "Prognose", fontsize=9, color="gray", style="italic")

plt.tight_layout()
plt.savefig("klimaprognose_2030.png", dpi=150, bbox_inches="tight")
print("\nDiagram lagret som 'klimaprognose_2030.png'")
plt.show()
