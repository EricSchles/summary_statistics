from scipy import stats
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd
import IPython
import random
import math

def summary(X, y):
    """
    Mutual information (MI) [R176] between two random variables is a non-negative value, which measures the dependency between the variables. 
    It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.
    """
    cols = X.columns.tolist()
    mi = mutual_info_regression(X, y)
    mi /= np.max(mi)
    f_test, f_test_pvals = f_regression(X, y)
    f_test /= np.max(f_test)
    t_test = [math.sqrt(elem) for elem in f_test]
    print(cols)
    print("F-test")
    print(list(f_test))
    print("F-test pvals")
    print(list(f_test_pvals))
    print("t-test")
    print(list(t_test))
    print("t-test pvals")
    print(list(f_test_pvals))
    print("mutual information")
    print(list(mi))
    
    
def test_summary():
    df = pd.DataFrame()
    df["A"] = np.array([random.randint(0,150) for _ in range(1000)])
    df["B"] = np.array([random.randint(0,150) for _ in range(1000)])
    df["y"] = np.array([random.randint(0,150) for _ in range(1000)])
    X = df[["A", "B"]]
    y = df["y"]
    summary(X, y)
    
test_summary()
