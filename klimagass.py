import pandas as pd
import statsmodels.api as sm

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

plot(aar, olje)
plot(aar, z1, "r")
plot(aar, z2, "b")
plot(aar, z3, "y")
plot(aar, z4, "g")
plot(aar, z5, "w")
plot(aar, z6, "p")
plot(aar, z7, "r")
plot(aar, z8, "r")
xlabel("År")
ylabel("Temperatur (grader C)")
show()