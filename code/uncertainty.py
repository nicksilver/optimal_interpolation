"""
Created on: Wed Aug  7 09:58:53 2013
Author: Nick Silverman
Email: nick.silverman11@gmail.com
Description: Class that calculates uncertainty in observations and model biases 
based on several different methods.
"""

import numpy as np                                                              # numerical python
import scipy.spatial                                                            # scipy spatial


class ObsUncertainty:
    """
    Calculates R matrix necessary for optimal interpolation and/or Kalman 
    filtering.
    """
    
    def __init__(self, obs_data, obs_mask, mod_data):
        """
        obs_data => observational values (yrs x n_obs)
        
        obs_ind => "1" represents obs location, "0" otherwise (n_cells x 1)
        
        mod_data => model values (yrs x ncols x nrows)
        """
        self.obs_data = obs_data
        self.obs_mask = obs_mask
        self.mod_data = mod_data
        self.yrs = self.mod_data.shape[0]
        self.n_obs = self.obs_data.shape[1]
        self.n_cells = mod_data.shape[1] 
        self.obs_ind = np.nonzero(obs_mask)
        
    # calculate mod-obs difference
    def mo_diff(self):
        mod_diff = np.zeros(self.obs_data.shape)
        for i in range(self.yrs):         
            mod_diff[i, :] = self.mod_data[i][self.obs_ind]
            mo_diff = mod_diff - self.obs_data
        return mo_diff
    
    # correlogram
    def correlogram(self, mo_diff):
        obs_ind_arr = np.array(self.obs_ind)
        corr = np.corrcoef(mo_diff.T)                                           # correlation matrix
        dist_con = scipy.spatial.distance.pdist(obs_ind_arr.T, 'euclidean')
        dist_mat = scipy.spatial.distance.squareform(dist_con)
        corr_flat = np.reshape(corr, (np.shape(corr)[0]**2, 1))
        dist_flat = np.reshape(dist_mat, (np.shape(dist_mat)[0]**2, 1))
        corr_gram = np.hstack((corr_flat, dist_flat))
        ind = np.nonzero(corr_gram[:, 1])
        corr_gram = corr_gram[ind]                                              # get rid of values with dist=0
        return corr_gram 
    
    # polyline regression fit to find nugget
    def poly_fit(self, corr_gram):    
        corr_sort = corr_gram[corr_gram[:, 1].argsort()]
        pcorr = np.polyfit(corr_sort[:,1], corr_sort[:,0], deg=2)
        p = np.poly1d(pcorr)
        p_nug = p(0)
        return p, p_nug                                                         # use value at lag=0
        
    def hollingsworth_lonnberg_obs(self):
        """
        This method has been adapted from Hollingsworth and Lonnberg (1986), 
        see also Tilmes (2001) for clear description of the method.  Any bias 
        in the observations or model should be removed before proceeding with 
        this method.
        
        In the method, a correlogram is developed using the difference between 
        observational values and model values for the grid cells in which they 
        both exist.  The nugget of the correlogram is then assumed to be the 
        spatially correlated part of the error which is attributed only to
        the model.  At this time, the nugget is assumed to be the fitted 
        polynomial where the lag distance equals zero.
        
        INPUTS:
        NONE
        
        OUTPUTS:
        R => observation covariance matrix
        """       
        
        mo_diff = self.mo_diff()                                                # mod-obs difference
        corr_gram = self.correlogram(mo_diff)                                   # correlogram
        p_nug = self.poly_fit(corr_gram)[1]                                     # find the nugget
        mo_var = np.mean(np.diag(np.cov(mo_diff.T)))                            # average variance of mod-obs
        Em = mo_var*p_nug                                                       # model uncertainty
        Eo = mo_var-Em                                                          # obs uncertainty
        R = np.dot(Eo, np.eye(self.n_obs))                                      # obs covariance matrices             
        return R
                            
    def lopez(self, doy_start, doy_end, sigma0=0.22, sigmaD=0.07, hemi=0):
        """
        This method is based on the work done by Lopez et al. 2011.  It has 
        been developed for flat terrain and grid-cells larger than 15km.  
        Nonetheless, it provides a basis for determining an approximate 
        estimate of gage representativity error (which is the most dominant 
        error).  Other errors (systematic and/or random) should be added to 
        this value.  
        
        The result will be a min, mean, and max value for the representativity 
        error accumulated over 4 months (2904 hours) using the statistics for 
        the values between the start and end doy. The representativity error is
        intended to be constant across all gages and no spatial correlation (R 
        matrix is diagonal only).
        
        :param doy_start: start day, day of year # (float, 0-365)
        :param doy_end: end day, day of year # (float, 0-365)
        :param sigma0: parameter (see Lopez et al. 2011 table 2)
        :param sigmaD: parameter (see Lopez et al. 2011 table 2)
        :param hemi: northern hemisphere=0, southern hemisphere=1
    
        Returns: R_min, R_mean, R_max (mm/winter).
        """  
        d_range = doy_end - doy_start+1
        r = np.zeros(d_range)
        for x in range(doy_start, doy_end+1):
            r[x-doy_start] = sigma0 + sigmaD*np.sin((np.pi/2)*((x-112.)/91.)+hemi*np.pi)
        return 2904*np.min(r), 2904*np.mean(r), 2904*np.max(r)
    
    def topo_var(self):
        pass


class ModUncertainty(ObsUncertainty):
    """
    Calculates P matrix necessary for optimal interpolation and/or Kalman 
    filtering.
    """
    
    def __init__(self, obs_data, obs_mask, mod_data):
        ObsUncertainty.__init__(self, obs_data, obs_mask, mod_data)
            
    def hollingsworth_lonnberg_mod(self):
        """
        This method has been adapted from Hollingsworth and Lonnberg (1986), 
        see also Tilmes (2001) for clear description of the method.  Any bias 
        in the observations or model should be removed before proceeding with 
        this method.
        
        In the method, a correlogram is developed using the difference between 
        observational values and model values for the grid cells in which they 
        both exist.  The nugget of the correlogram is then assumed to be the 
        spatially correlated part of the error which is attributed only to
        the model.  At this time, the nugget is assumed to be the fitted 
        polynomial where the lag distance equals zero.
        
        INPUTS:
        NONE
        
        OUTPUTS:
        P => model covariance matrix
        """ 
        
        mo_diff = self.mo_diff()                                                # mod-obs difference
        corr_gram = self.correlogram(mo_diff)                                   # correlogram
        p, p_nug = self.poly_fit(corr_gram)                                     # polyfit and nugget
        mo_var = np.mean(np.diag(np.cov(mo_diff.T)))                            # average variance of mod-obs
        Em = mo_var*p_nug                                                       # model uncertainty
        
        ind_arr = np.ones((self.dom_rows, self.dom_cols))                       # complete dist. mat.
        ind_all = np.array(np.nonzero(ind_arr))
        dist_all = scipy.spatial.distance.pdist(ind_all.T, 'euclidean')
        dist_all_mat = scipy.spatial.distance.squareform(dist_all)
        dist_all_flat = np.reshape(dist_all_mat, (self.n_cells**2, 1)) 
        pm = np.reshape(p(dist_all_flat)/p_nug, (self.n_cells, self.n_cells))
        P = np.dot(Em, pm) 
        return P
    
    
    
    
    
    