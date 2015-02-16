"""
By: Nick Silverman
Date: 7/25/2014
Description: This script provides bootstrap function for spatially distributed 
climate data. 
"""

from random import randint
import numpy as np

def bootstrap(clim_data, n):
    """
    Returns an array of n bootstrapped (with replacement) means. 
    
    Args:
        clim_data (numpy array): spatially distributed climate data
        
        n (int): number of boostrapped samples 
    """
    rows = clim_data.shape[0]
    cols = clim_data.shape[1]
    boot_data = np.zeros([n, cols])
    for i in range(n):   
        samp = np.zeros([rows, cols])
        for j in range(rows):
            k = randint(0, rows-1)
            samp[j,:] = clim_data[k,:]
            samp_avg = samp.mean(axis=0)
        boot_data[i,:] = samp_avg
    return boot_data
