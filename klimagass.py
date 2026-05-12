import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

klima = pd.read_csv("figur-1-totalt-klimagass.csv", sep=";", decimal=",")

aar = klima.iloc[:, 0]
olje = klima.iloc[:, 1]
industri = klima.iloc[:, 2]
veitrafikk = klima.iloc[:, 5]
motorfart = klima.iloc[:, 6]

lowess = sm.nonparametric.lowess
z1 = lowess(olje, aar, frac = 0.2, return_sorted = False)
z2 = lowess(industri, aar, frac = 0.2, return_sorted = False)
z5 = lowess(veitrafikk, aar, frac = 0.2, return_sorted = False)
z6 = lowess(motorfart, aar, frac = 0.2, return_sorted = False)

plt.plot(aar, olje)
plt.plot(aar, z1, "r")
plt.plot(aar, z2, "b")
plt.plot(aar, z5, "w")
plt.plot(aar, z6, "p")
plt.xlabel("År")
plt.ylabel("Temperatur (grader C)")
plt.show()