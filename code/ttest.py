"""
Created on: Fri Oct 18 14:09:55 2013
Author: Nick Silverman
Email: nick.silverman11@gmail.com

Description: A series of functions that help perform T-Tests on two 
populations using R statistics.
"""

import rpy2.robjects as R
import numpy as np
import math
import scipy.stats as stats

def mm_ttest(x1, x2, v1, v2, n1, n2, mu=0):
    """
    Performs an independent Student t test between the mean of two
    samples with unequal 
    variances and arbitrary sample sizes. This test is identical to the 
    scipy implementation except that it permits to provide the sample
    variances and sizes directly  
    
    Returns: t, p
    t: quantile of the t distribution
    p: probability of t
    """
    
    if (v1==v2): #if samples have equal variance
        df = n1 + n2 - 2
        svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / float(df)
        denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    else:
        vn1 = v1 / n1
        vn2 = v2 / n2
        df = ((vn1 + vn2)**2) / ((vn1**2) / (n1 - 1) + (vn2**2) / (n2 - 1))

        # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
        # Hence it doesn't matter what df is as long as it's not NaN.
        df = np.where(np.isnan(df), 1, df)
        denom = np.sqrt(vn1 + vn2)

    d = x1 - x2 - mu
    t = np.divide(d, denom)
    t, prob = stats._ttest_finish(df, t)

    return t, prob

def apply_mm_ttest(py_data1, py_data2, mu, rho=0.05, alt="two.sided"):
    """
    Performs a paired t-test using the R t.test function
    
    :param py_data1: numpy array for sample 1
    :param py_data2: numpy array for sample 2
    :param mu: null hypothesis value (default = 0) 
    :param rho: signifcance threshold for p-value
    :param alt: whether to perform a "two.sided", "greater", "less" t-test. 
    See description in R help for t.test function alternative.
    
    :return: binary dataframe where "1" indicates significance and "0"
    indicates insignificant.
    """
    
    result = np.zeros((py_data1.shape[1]))
    for x in range(py_data1.shape[1]):
        pval = r_ttest(py_data1[:,x], py_data2[:,x], mu[x], alt)
        result[x] = pval
    
    sig = np.zeros((result.size))    
    ind = result<=rho
    sig[ind] = 1
    return sig

def equal_data(py_data1, py_data2):
    """
    Makes sure that the data has identical dimensions. If dimensions are 
    different then the function will add the appropriate number or rows filled
    with "nan".
    
    :param py_data1: numpy array for sample 1
    :param py_data2: numpy array for sample 2
    
    :return: updated py_data1, updated py_data2
    """
    
    size1 = py_data1.shape[0]
    size2 = py_data2.shape[0]
    diff = np.abs(size1-size2)
    
    if size1>size2:
        for t in range(diff):
            py_data2 = np.hstack((py_data2, np.nan))
        
    elif size1<size2:
        for t in range(diff):
            py_data1 = np.hstack((py_data1, np.nan))
    
    return py_data1, py_data2


def r_ttest(py_data1, py_data2, mu=0, alt="two.sided"):
    """
    Defined for use within apply_ttest function.
    """
    
    # Make sure datasets have equal dimensions
    py_data1, py_data2 = equal_data(py_data1, py_data2)
    
    # Bring in t.test function from R
    t_test = R.r['t.test']    
    
    # Convert numpy array to R vector for both datasets
    data1 = R.FloatVector(py_data1)
    data2 = R.FloatVector(py_data2)
    
    # Perform t-test on two populations
    test = t_test(data1, data2, mu=mu, 
                  **{'paired':True, 'na.action':R.StrVector(("na.exclude",)), 
                     'alternative':R.StrVector((alt,))})
    
    # Index names from R list and report the p-value                 
    names = test.names
    pval = test[names.index('p.value')][0]
    return pval
    
    
def apply_ttest(py_data1, py_data2, mu, rho=0.05, alt="two.sided"):
    """
    Performs a paired t-test using the R t.test function
    
    :param py_data1: numpy array for sample 1
    :param py_data2: numpy array for sample 2
    :param mu: null hypothesis value (default = 0) 
    :param rho: signifcance threshold for p-value
    :param alt: whether to perform a "two.sided", "greater", "less" t-test. 
    See description in R help for t.test function alternative.
    
    :return: binary dataframe where "1" indicates significance and "0"
    indicates insignificant.
    """
    
    result = np.zeros((py_data1.shape[1]))
    for x in range(py_data1.shape[1]):
        pval = r_ttest(py_data1[:,x], py_data2[:,x], mu[x], alt)
        result[x] = pval
    
    sig = np.zeros((result.size))    
    ind = result<=rho
    sig[ind] = 1
    return sig
    
  
def r_invttest(py_data, mu=0, df=8, p=0.95):
    """
    Defined for use within apply_invttest function.
    """
    
    # Bring in qt function from R    
    qt_test = R.r['qt']
    
    # Calculate standard error
    err = py_data.std()/math.sqrt(len(py_data))
    
    # Perform inverted t-test
    val = qt_test(p, df)*err+mu  
    return val[0]
    

def apply_invttest(py_data, mu, df=8, p=0.95):
    """
    Calculates the value to give significance in a student t-test using the R 
    qt function.
    
    :param py_data: numpy array of data
    :param mu: null hypothesis value (default = 0) 
    :param df: degrees of freedom from t-test of py_data
    :param p: probability (default = 0.95)
    See description in R help for qt function alternative.
    
    :return: vector of minimum values to achieve significance using
    t-distribution
    """
    
    result = np.zeros((py_data.shape[1]))
    for x in range(py_data.shape[1]):
        val = r_invttest(py_data[:,x], mu=mu[x], df=df, p=p)
        result[x] = val
    return result
    
    
def sigcoords(xy, sig):
    """
    Combines binary significance data with xy grid and then eliminates all rows
    with zeros, thus leaving an array of x, y coordinates for significant 
    values.
    
    :param xy: xy coordinates
    :param sig: array of zeros and ones, ones represent significance
    
    :return: coordinates of significant values
    """
    sig_data = np.vstack((xy[:, 0], xy[:, 1], sig))
    sig_xy = sig_data[:, sig_data.all(axis=0)][:2, :] 
    return sig_xy

def gausstest(data, sig=2):
    """
    Performs an Anderson-Darling test for gaussianity on a 2D numpy array

    :param data: 2D numpy array. First dimension is time, second dimension is space.

    :param sig: Index for significance level given in scipy.stats.anderson function (i.e. 5% = 2). Default is '2' or 5%.

    :return: A 2D numpy boolean array the shape of 'data'. The number one means the distribution likely came from a
    normal distribution. The number zero rejects this hypothesis.
    """
    res = np.zeros((data.shape[1]))
    for i in range(data.shape[1]):
        s, v, p = stats.anderson(data[:, i], 'norm')
        if s < v[sig]:
            r = 1
        else:
            r = 0
        res[i] = r
    return res
