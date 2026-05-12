import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

klima = pd.read_csv("figur-1-totalt-klimagass.csv", sep=";", decimal=",")

aar = klima.iloc[:, 0]
olje = klima.iloc[:, 1]
industri = klima.iloc[:, 2]
energi = klima.iloc[:, 3]
oppvarming = klima.iloc[:, 4]
veitrafikk = klima.iloc[:, 5]
motorfart = klima.iloc[:, 6]
jordbruk = klima.iloc[:, 7]
andre = klima.iloc[:, 8]

lowess = sm.nonparametric.lowess
z1 = lowess(olje, aar, frac = 0.2, return_sorted = False)
z2 = lowess(industri, aar, frac = 0.2, return_sorted = False)
z3 = lowess(energi, aar, frac = 0.2, return_sorted = False)
z4 = lowess(oppvarming, aar, frac = 0.2, return_sorted = False)
z5 = lowess(veitrafikk, aar, frac = 0.2, return_sorted = False)
z6 = lowess(motorfart, aar, frac = 0.2, return_sorted = False)
z7 = lowess(jordbruk, aar, frac = 0.2, return_sorted = False)
z8 = lowess(andre, aar, frac = 0.2, return_sorted = False)

plt.plot(aar, olje)
plt.plot(aar, z1, "r")
plt.plot(aar, z2, "b")
plt.plot(aar, z3, "y")
plt.plot(aar, z4, "g")
plt.plot(aar, z5, "w")
plt.plot(aar, z6, "p")
plt.plot(aar, z7, "r")
plt.plot(aar, z8, "r")
plt.xlabel("År")
plt.ylabel("Temperatur (grader C)")
plt.show()