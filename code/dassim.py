import numpy as np

def H_mat(obs_mask):
    """
    Creates H transformation matrix of zeros and ones based on a single vector 
    of zeros (no station) and ones (station location)
    
    :param obs_mask: Binary mask vector (1=station, 0=no station)
    
    Returns: H matrix with dimensions ncells x nobs    
    """
    H = [ ]
    for i in np.flatnonzero(obs_mask):
        h_temp = np.zeros(obs_mask.shape[0])
        h_temp[i] = 1
        H.append(h_temp)
    H = np.array(H)
    return H

def kalman_K(P, H, R):
    """
    Calculates the Kalman weight matrix.
    
    :param P: Model covariance matrix
    :param H: Obs transformation matrix
    :param R: Obs covariance matrix
    
    Returns: Kalman weight matrix (K)
    """
    num = np.dot(P, H.T)
    den = np.dot(np.dot(H, P), H.T) + R
    K = np.dot(num, np.linalg.inv(den))
    return K

def opt_interp(mod, H, K, obs):
    """
    Calculates updated model values.
    
    :param mod: Model values
    :param H: Obs transformation matrix
    :param b_plus: Updated bias values
    :param K: Kalman weight matrix
    :param obs: Observation values
    
    Returns: Updated model values in a ndarray.
    """
    Hx = np.dot(H, mod)
    zHx = obs - Hx
    KzHx = np.dot(K, zHx)
    x_plus = mod + KzHx
    return x_plus

def update_P(K, H, P):
    """
    Calculates the updated covariance matrix for the model.
    
    :param K: Kalman weight matrix
    :param H: Obs transformation matrix
    :param P: Initial model covariance matrix
    
    Returns: Updated covariance matrix for the model (P).
    """
    KH = np.dot(K, H)
    P_plus = P - np.dot(KH, P)
    return P_plus
