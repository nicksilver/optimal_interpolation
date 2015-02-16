"""
Created on: Wed Aug  7 09:58:53 2013
Author: Nick Silverman
Email: nick.silverman11@gmail.com
Description: Driver file to run uncertainty analysis
"""

#==============================================================================
# Import modules
#%%============================================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import ttest
import os
import geotiff
import uncertainty
import dassim
<<<<<<< HEAD
#import bootclim
=======
import bootclim
>>>>>>> 3d77d056e94f4995e65996532654f6eff50f13f8

#==============================================================================
# Bring in data
#%%============================================================================
ncells = 8639
nobs = 58
HOME = os.path.expanduser("~/")
<<<<<<< HEAD
data_path = HOME+"Copy/workspace/bayes_precip/data/"
=======
data_path = "./"#HOME+"Copy/workspace/bayes_precip/data/"
>>>>>>> 3d77d056e94f4995e65996532654f6eff50f13f8

gfs_data = np.loadtxt(data_path+"win_gfs.asc")
pcm_h_data = np.loadtxt(data_path+"PR_win_pcm_hist.asc")
#pcm_h_boot = bootclim.bootstrap(pcm_h_data, 100)
<<<<<<< HEAD
pcm_f_data = np.loadtxt(data_path+"PR_win_pcm_fut.asc")[0:6, :]  # pick the first 6 years of the data
#pcm_f_boot = bootclim.bootstrap(pcm_f_data, 100)
obs_data = np.loadtxt(data_path+"snotelPrec_matrix.txt").T

obs_mask = np.loadtxt(data_path+"snotel_flag.txt")
xyz = np.loadtxt(data_path+'ascGrid.xyz', skiprows=1)[:, 0:2]
=======
pcm_f_data = np.loadtxt(data_path+"PR_win_pcm_fut.asc")[0:6,:]  # pick the first 6 years of the data
#pcm_f_boot = bootclim.bootstrap(pcm_f_data, 100)
#obs_data = np.loadtxt(data_path+"snotelPrec_matrix.txt")[:,:-4].T
obs_data = np.loadtxt(data_path+"snotelPrec_matrix.txt").T

obs_mask = np.loadtxt(data_path+"snotel_flag.txt")
xyz = np.loadtxt(data_path+'ascGrid.xyz', skiprows=1)[:,0:2]
>>>>>>> 3d77d056e94f4995e65996532654f6eff50f13f8
dem = np.loadtxt(data_path+"dem.txt", skiprows=5)

#==============================================================================
# Matrices for optimal interpolation
#%%============================================================================
doy_start = 1  # Jan. 1st
<<<<<<< HEAD
doy_end = 120  # Apr. 30th (extra month to compensate for not including december)
unc_obs = uncertainty.ObsUncertainty(obs_data, obs_mask, gfs_data)  # obs uncertainty object
sig_min, sig_mean, sig_max = unc_obs.lopez(doy_start, doy_end)  # representativity error
R_min = 1*sig_min**2*np.diag(np.ones(nobs))
R_mean = 2*sig_mean**2*np.diag(np.ones(nobs))
R_max = 160000*sig_max**2*np.diag(np.ones(nobs))  # large number worst case scenario
=======
doy_end = 120  # Apr. 30th
unc_obs = uncertainty.ObsUncertainty(obs_data, obs_mask, gfs_data)  # obs uncertainty object
sig_min, sig_mean, sig_max = unc_obs.lopez(doy_start, doy_end)  # rep. error (dates go through april to compensate for not including december)
R_min = 1*sig_min**2*np.diag(np.ones(nobs))
R_mean = 2*sig_mean**2*np.diag(np.ones(nobs))
R_max = 160000*sig_max**2*np.diag(np.ones(nobs)) #large number worst case scenario
>>>>>>> 3d77d056e94f4995e65996532654f6eff50f13f8
P_cov = 1.2*np.cov(gfs_data.T)  # mod cov directly from data 
Unc = np.sqrt(np.diag(P_cov))
X = np.mean(gfs_data, axis=0).reshape(ncells, 1)  # model matrix
Z = np.mean(obs_data, axis=0).T.reshape(np.sum(obs_mask), 1)  # obs matrix
P = P_cov  # initial mod cov matrix
H = dassim.H_mat(obs_mask)

#==============================================================================
# Optimal Interpolation
#%%============================================================================
<<<<<<< HEAD
R = R_min  # + np.diag(0.25*Z[:,0])**2  # set R matrix
=======
R = R_min #+ np.diag(0.25*Z[:,0])**2  # set R matrix
>>>>>>> 3d77d056e94f4995e65996532654f6eff50f13f8
K = dassim.kalman_K(P, H, R)
X_plus = dassim.opt_interp(X, H, K, Z)
P_plus = dassim.update_P(K, H, P)
Unc_plus = np.sqrt(np.diag(P_plus))
#np.save("stdev_post_min", Unc_plus)
#P_plus = np.load("p_post_mean.npy")
print "stdev_post_min written..."

R = R_mean #+ np.diag(0.25*Z[:,0])**2  # set R matrix
K = dassim.kalman_K(P, H, R)
X_plus = dassim.opt_interp(X, H, K, Z)
P_plus = dassim.update_P(K, H, P)
Unc_plus = np.sqrt(np.diag(P_plus))
#np.save("stdev_post_mean", Unc_plus)
#P_plus = np.load("p_post_mean.npy")
print "stdev_post_mean written..."


R = R_max #+ np.diag(0.25*Z[:,0])**2  # set R matrix
K = dassim.kalman_K(P, H, R)
X_plus = dassim.opt_interp(X, H, K, Z)
P_plus = dassim.update_P(K, H, P)
Unc_plus = np.sqrt(np.diag(P_plus))
#np.save("stdev_post_max", Unc_plus)
#P_plus = np.load("p_post_max.npy")
print "stdev_post_max written..."


#==============================================================================
# Load mean error of precipitation estimates
#%%============================================================================
Unc_plus_mean = np.load("stdev_post_mean.npy")
est_var_mean =  Unc_plus_mean * Unc_plus_mean

Unc_plus_min = np.load("stdev_post_min.npy")
est_var_min =  Unc_plus_min * Unc_plus_min

Unc_plus_max = np.load("stdev_post_max.npy")
est_var_max =  Unc_plus_max * Unc_plus_max
#==============================================================================
# T-tests
#%%============================================================================
mu_0 = np.zeros((pcm_h_data.shape[1]))
#mu_unc_mean = 2*Unc_plus_mean*1.0  # uncertainty is ~2X stdev
#mu_unc_min = 2*Unc_plus_mean*0.75  # uncertainty is ~2X stdev
#mu_unc_max = 2*Unc_plus_mean*1.25# uncertainty is ~2X stdev

v_h = np.var(pcm_h_data, axis=0, ddof=1)
v_f = np.var(pcm_f_data, axis=0, ddof=1)
mean_h = np.mean(pcm_h_data, axis=0)
mean_f = np.mean(pcm_f_data, axis=0)

mu_unc_min = (1 - np.sqrt(v_h+v_f)/np.sqrt(v_h+v_f+2.2*est_var_min)) *(mean_f - mean_h)
print mu_unc_min

mu_unc_mean = (1 - np.sqrt(v_h+v_f)/np.sqrt(v_h+v_f+2.2*est_var_mean)) *(mean_f - mean_h)
print mu_unc_mean

mu_unc_max = (1 - np.sqrt(v_h+v_f)/np.sqrt(v_h+v_f+2.2*est_var_max)) *(mean_f - mean_h)
print mu_unc_max

sig = ttest.apply_ttest(pcm_f_data, pcm_h_data, mu=mu_0, rho=0.05,
                        alt="two.sided")  # perform t-test actual
sig_unc_mean = ttest.apply_ttest(pcm_f_data, pcm_h_data, mu=mu_unc_mean, rho=0.05,
                                 alt="two.sided")  # perform t-test with uncert.
sig_unc_min = ttest.apply_ttest(pcm_f_data, pcm_h_data, mu=mu_unc_min, rho=0.05,
                                alt="two.sided")  # perform t-test with uncert.
sig_unc_max = ttest.apply_ttest(pcm_f_data, pcm_h_data, mu=mu_unc_max, rho=0.05,
                                alt="two.sided")  # perform t-test with uncert. 
<<<<<<< HEAD
#inv_vals = ttest.apply_invttest(gfs_data, mu=mu_0, df=8, p=0.95) # inverted t-test
#inv_vals_unc = ttest.apply_invttest(gfs_data, mu=mu_unc_mean, df=8, p=0.95) # inverted t-test with uncertainty added
=======
inv_vals = ttest.apply_invttest(gfs_data, mu=mu_0, df=8, p=0.95) # inverted t-test
inv_vals_unc = ttest.apply_invttest(gfs_data, mu=mu_unc_mean, df=8, p=0.95) # inverted t-test with uncertainty added
>>>>>>> 3d77d056e94f4995e65996532654f6eff50f13f8

unsig_unc_mean = mu_0
unsig_unc_min = mu_0
unsig_unc_max = mu_0
unsig_unc_mean[(sig+sig_unc_mean)==1] = 1
unsig_unc_min[(sig+sig_unc_min)==1] = 1
unsig_unc_max[(sig+sig_unc_max)==1] = 1
xy_sig_coord = ttest.sigcoords(xyz, sig)
xy_unc_coord_mean = ttest.sigcoords(xyz, sig_unc_mean)
xy_unc_coord_min = ttest.sigcoords(xyz, sig_unc_min)
xy_unc_coord_max = ttest.sigcoords(xyz, sig_unc_max)
xy_unsig_coord_mean = ttest.sigcoords(xyz, unsig_unc_mean)
xy_unsig_coord_min = ttest.sigcoords(xyz, unsig_unc_min)
xy_unsig_coord_max = ttest.sigcoords(xyz, unsig_unc_max)
obs_coord = ttest.sigcoords(xyz, obs_mask)
diff = 100*(Unc - Unc_plus_mean)/Unc #np.mean(pcm_f_data, axis=0) - np.mean(pcm_h_data, axis=0)

#%%============================================================================
# Gaussianity tests   
#==============================================================================
res_wrf = ttest.gausstest(gfs_data)
res_snot = ttest.gausstest(obs_data)

#%%============================================================================
# Save coordinate files    
#==============================================================================
#np.savetxt('unc_coords.csv', xy_unc_coord_mean.T)
#np.savetxt('sig_coords.csv', xy_sig_coord.T)
#np.savetxt('unsig_coords.csv', xy_unsig_coord_mean.T)

'''
Need to use QGIS to calculate aspect and elev of the different coordinate sets.
Import the above .csv file on top of aspect and DEM rasters.  The use the point
sampling tool to extract the aspect of elevation at each point.
'''

#%%==============================================================================
# Plot    
#==============================================================================
cen_lat = 47
cen_lon = -114.25
truelat1 = 30.0
truelat2 = 60.0
standlon = -114.0
width_meters = 100*4000
height_meters = 120*4000

# Create basemap
m = Basemap(resolution='i', projection='lcc', width=width_meters,
            height=height_meters, lat_0=cen_lat, lon_0=cen_lon, lat_1=truelat1,
            lat_2=truelat2)

sig_xy = xy_sig_coord  # set sig points to plot
sig_xy_unc_mean = xy_unc_coord_mean
sig_xy_unc_min = xy_unc_coord_min
sig_xy_unc_max = xy_unc_coord_max
unsig_xy_mean = xy_unsig_coord_mean
unsig_xy_min = xy_unsig_coord_min
unsig_xy_max = xy_unsig_coord_max
obs_xy = obs_coord

# Gridded data
x = xyz[:,0]
xi = np.linspace(x.min(), x.max(), 200)
y = xyz[:,1]
yi = np.linspace(y.min(), y.max(), 200)
X,Y = np.meshgrid(xi, yi)
diff_grid = plt.mlab.griddata(x, y, diff, xi, yi)
lat, lon = m(X, Y)
lat_1, lon_1 = m(x, y)

# Background image (see Jared's basemap_example.py)
res = 500.
dsElev = geotiff.RasterDataset(data_path+'basemap_data/hillshade1.tif')
latElev, lonElev = dsElev.getCoordGrid1d()
latElev = np.sort(latElev)
nx = int((m.xmax-m.xmin)/res)+1; ny = int((m.ymax-m.ymin)/res)+1
elev = dsElev.readAsArray()
elev = np.flipud(elev)
elev = m.transform_scalar(elev, lonElev, latElev, nx, ny)

# Plot basemap
m.drawstates(linewidth=1, zorder=7)
m.drawcountries(linewidth=1, zorder=6)
m.imshow(elev, cmap=plt.cm.gray)
p = m.contourf(lat, lon, diff_grid, cmap=plt.cm.rainbow, alpha=0.3,
               antialiased=True, zorder=5, levels=np.linspace(0,64,9))  # use cm.rainbow for diff, cm.gray for grayscale, cm.gist_ncar for precip, cm.hot for variance, cm.gist_earth for invttest
p = m.contourf(lat, lon, diff_grid, cmap=plt.cm.rainbow, alpha=0.3,
               antialiased=True, zorder=5, levels=np.linspace(0,64,9))
cbar = m.colorbar(p, location='right', pad="5%")

cbar.set_alpha(1)
cbar.draw_all()
cbar.set_label("percent difference")
sig_x, sig_y = m(sig_xy[0,:], sig_xy[1,:])
sig_x_unc_mean, sig_y_unc_mean = m(sig_xy_unc_mean[0,:], sig_xy_unc_mean[1,:])
sig_x_unc_min, sig_y_unc_min = m(sig_xy_unc_min[0,:], sig_xy_unc_min[1,:])
sig_x_unc_max, sig_y_unc_max = m(sig_xy_unc_max[0,:], sig_xy_unc_max[1,:])
#unsig_x_mean, unsig_y_mean = m(unsig_xy_mean[0,:], unsig_xy_mean[1,:])
#unsig_x_min, unsig_y_min = m(unsig_xy_min[0,:], unsig_xy_min[1,:])
#unsig_x_max, unsig_y_max = m(unsig_xy_max[0,:], unsig_xy_max[1,:])
#obs_x, obs_y = m(obs_xy[0,:], obs_xy[1,:])
m0 = m.scatter(sig_x, sig_y, s=1, marker='o', color='black', alpha=1., zorder=6)
#m.scatter(sig_x_unc_mean, sig_y_unc_mean, s=1, marker='o', color='black', zorder=8, edgecolor='k')
#m.scatter(unsig_x, unsig_y, s=1, marker='o', color='black', alpha=.3, zorder=8)
#m.scatter(obs_x, obs_y, s=30, marker='^', color='white', edgecolor='k', zorder=9)
m1 = m.scatter(sig_x_unc_min, sig_y_unc_min, s=15, marker='o', color='#00CC00', zorder=7, edgecolor='k')
m2 = m.scatter(sig_x_unc_mean, sig_y_unc_mean, s=15, marker='o', color='#FFFF00', zorder=8, edgecolor='k')
m3 = m.scatter(sig_x_unc_max, sig_y_unc_max, s=15, marker='o', color='#FF3300', zorder=9, edgecolor='k')
#labels = ['significant',r'detectable, best case', r'detectable', r'detectable worst case']
#plt.legend([m0, m1, m2, m3], labels, loc=4)

#plt.savefig('Detectability.png')
plt.show()

#%%============================================================================
# Plot covariance matrix
#==============================================================================
#im = plt.imshow(P[0:300,0:300], origin='lower', interpolation='nearest')
#plt.colorbar(im)
#plt.show()
