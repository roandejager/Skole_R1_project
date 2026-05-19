import math
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

klima = pd.read_csv("figur-1-totalt-klimagass.csv", sep=";", decimal=",")

aar = klima.iloc[:, 0]
olje = klima.iloc[:, 1]
industri = klima.iloc[:, 2]
veitrafikk = klima.iloc[:, 5]
motorfart = klima.iloc[:, 6]

lowess = sm.nonparametric.lowess
z1 = lowess(olje, aar, frac=0.2, return_sorted=False)
z2 = lowess(industri, aar, frac=0.2, return_sorted=False)
z3 = lowess(veitrafikk, aar, frac=0.2, return_sorted=False)
z4 = lowess(motorfart, aar, frac=0.2, return_sorted=False)

n = len(aar)
o_snitt = sum(olje) / n
i_snitt = sum(industri) / n
v_snitt = sum(veitrafikk) / n
m_snitt = sum(motorfart) / n

o_s = math.sqrt(sum((olje - o_snitt) ** 2) / (n - 1))
i_s = math.sqrt(sum((industri - i_snitt) ** 2) / (n - 1))
v_s = math.sqrt(sum((veitrafikk - v_snitt) ** 2) / (n - 1))
m_s = math.sqrt(sum((motorfart - m_snitt) ** 2) / (n - 1))

o_se = o_s / math.sqrt(n)
i_se = i_s / math.sqrt(n)
v_se = v_s / math.sqrt(n)
m_se = m_s / math.sqrt(n)

print(f'Gjennomsnittet for oljens CO2-utslipp gjennom årene er {round(o_snitt, 4)}, standardavvik {round(o_s, 4)} og standardfeil {round(o_se, 4)}')
print(f'Gjennomsnittet for industriens CO2-utslipp gjennom årene er {round(i_snitt, 4)}, standardavvik {round(i_s, 4)} og standardfeil {round(i_se, 4)}')
print(f'Gjennomsnittet for veitrafikkens CO2-utslipp gjennom årene er {round(v_snitt, 4)}, standardavvik {round(v_s, 4)} og standardfeil {round(v_se, 4)}')
print(f'Gjennomsnittet for motortrafikkens CO2-utslipp gjennom årene er {round(m_snitt, 4)}, standardavvik {round(m_s, 4)} og standardfeil {round(m_se, 4)}')

plt.plot(aar, z1, "r", label="Olje og gass")
plt.plot(aar, z2, "b", label="Industri")
plt.plot(aar, z3, "g", label="Veitrafikk")
plt.plot(aar, z4, "m", label="Motorfart")

plt.plot(aar, olje, "r")
plt.plot(aar, industri, "b")
plt.plot(aar, veitrafikk, "g")
plt.plot(aar, motorfart, "m")

plt.xlabel("År")
plt.ylabel("Million tonn CO2")
plt.legend(loc="upper right")
plt.grid()

plt.show()
