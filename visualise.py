
import matplotlib.pyplot as plt
import numpy, statistics
import pandas as pd
import statsmodels.api as sm

rm = 0.000622796
rf = 6.85951987 / (252 * 100)


def readColumn(sname, cname):
    df = pd.read_excel("dataset.xlsx", sheet_name = sname)
    raw = df[cname].tolist()
    return [x for x in raw if not pd.isna(x)]

def calcValuesFor(sname):
    print("--------------------- " + sname + " ---------------------")
    nav = readColumn(sname, "NAV")
    bmi = readColumn(sname, "BMI")
    hpr = readColumn(sname, "HPR")
    er = readColumn(sname, "ER")
    scatterdata = (nav, bmi)
    regrdata = (hpr, er)
    print(" > Generating Scatter Plot for NAV x BMI.")
    scatter_plot = scatterWithTrend(scatterdata, ("NAV", "BMI", sname))
    print(" > Scatter Plot Generated.")
    print(" > Generating Linear Regression.")
    regression_plot = calculateRegressionConsts(regrdata)
    print(" > Linear Regression Generated.")
    print(" > P-Value: ", regression_plot[0])
    b = regression_plot[1]
    print(" > Beta-Value: ", b)
    rp = float(sum(er)/len(er))
    print(" > Average Expected Returns: ", rp)
    stddev = statistics.pstdev(er)
    print(" > Standard Deviation (Expected Returns): ", stddev)
    jalpha = rp - (rf + b * (rm - rf))
    sharpes = (rp * 100 - rf) / (stddev * 100)
    treynors = (rp * 100 - rf) / b
    print(" > Jensen's Alpha: ", jalpha * 252 * 100, "%")
    print(" > Sharpe's Ratio: ", sharpes)
    print(" > Treynor's Ratio: ", treynors)
    scatter_plot.show()



def scatterWithTrend(vertices, titles=None, annotate=False):
    x, y = vertices
    plt.plot(x, y, 'o')
    z = numpy.polyfit(x, y, 1)
    p = numpy.poly1d(z)
    plt.plot(x, p(x), "r--")
    print(" > Trend Line: y = %.6f x + (%.6f) " % (z[0],z[1]))
    if titles is not None:
        if(len(titles) == 3):
            plt.title(titles[2])
        plt.xlabel(titles[0])
        plt.ylabel(titles[1])
    if annotate:
        for i in range(len(x)):
            plt.annotate("({}, {})".format(x[i], y[i]), (x[i], y[i]))
    return plt

def calculateRegressionConsts(dataset):
    x, y = dataset
    x2 = sm.add_constant(x)
    est = sm.OLS(y, x2)
    res = est.fit()
    return (res.pvalues[0], res.params[1])

if __name__ == "__main__":
    xl = pd.ExcelFile("dataset.xlsx")
    worksheets = xl.sheet_names
    for worksheet in worksheets[:-1]:
        calcValuesFor(worksheet)
