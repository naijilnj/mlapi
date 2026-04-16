# Linear Regression

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([10, 20, 30, 40, 50, 60, 65, 70, 80, 90])

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))
print(mymodel[:5])

print("Correlation coefficient:", r)

yhat = myfunc(10)
print("Predicted value:", yhat)

plt.scatter(x, y)
plt.scatter(10, yhat, color='red')
plt.plot(x, mymodel)

plt.xlabel('x')
plt.ylabel('y')
plt.title("Linear Regression")

plt.show()



# Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y = [100, 90, 80, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 90, 97, 99, 100]

mymodel = np.poly1d(np.polyfit(x, y, 3))

yhat = mymodel(10.5)
print("Predicted value:", yhat)

plt.scatter(x, y)
plt.scatter(10.5, yhat, color='red')
plt.plot(x, mymodel(x))

plt.xlabel('x')
plt.ylabel('y')
plt.title("Polynomial Regression")

plt.show()