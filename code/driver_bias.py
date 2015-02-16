"""
Created on: Wed Aug  7 09:58:53 2013
Author: Nick Silverman
Email: nick.silverman11@gmail.com
Description: Driver file to run bias analysis
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

#%%============================================================================
# Bring in data
#%%============================================================================
yrs = 9
ncells = 8639
HOME = os.path.expanduser("~/")
data_path = HOME+"/SparkleShare/workspace/hist-futu_comparison/data/"

gfs_data = np.loadtxt(data_path+"win_gfs.asc")
pcm_h_data = np.loadtxt(data_path+"PR_win_pcm_hist_corr.asc")
pcm_f_data = np.loadtxt(data_path+"PR_win_pcm_fut_corr.asc")
obs_data = np.loadtxt(data_path+"snotelPrec_matrix.txt")[:,0:9].T
obs_mask = np.loadtxt(data_path+"snotel_flag.txt")
xyz = np.loadtxt(data_path+'ascGrid.xyz', skiprows=1)[:,0:2]
dem = np.loadtxt(data_path+"dem.txt", skiprows=5)

#%%============================================================================
# Matrices for optimal interpolation
#%%============================================================================
bias = 0.20                                                                      # percent obs undercatch
unc_obs = uncertainty.ObsUncertainty(obs_data, obs_mask, gfs_data)              # obs uncertainty
R = 100.*unc_obs.hollingsworth_lonnberg_obs()                                   # obs covariance matrix
#unc_mod = uncertainty.ModUncertainty(obs_data, obs_mask, gfs_data)             # model uncertainty
# P_hl = unc_mod.hollingsworth_lonnberg_mod()                                   # mod cov from H-L method
P_cov = np.cov(gfs_data.T)                                                      # mod cov directly from data 
Unc = np.sqrt(np.diag(P_cov))                  
X = np.mean(gfs_data, axis=0).reshape(ncells, 1)                                # model matrix
Z = np.mean(obs_data, axis=0).T.reshape(np.sum(obs_mask), 1)*(1.+bias)          # obs matrix
P = P_cov                                                                       # initial mod cov matrix
H = dassim.H_mat(obs_mask)

#%%============================================================================
# Optimal Interpolation
#%%============================================================================
K = dassim.kalman_K(P, H, R)
X_plus = dassim.opt_interp(X, H, K, Z)
P_plus = dassim.update_P(K, H, P_cov)
Unc_plus = np.sqrt(np.diag(P_plus))

x_diff = X-X_plus
obs_xy = ttest.sigcoords(xyz, obs_mask)

#%%============================================================================
# Plot    
#%%============================================================================
#cen_lat = 46.94521
#cen_lon = -113.3452
cen_lat = 47
cen_lon = -114.25
truelat1 = 30.0
truelat2 = 60.0
standlon = -114.0
width_meters = 100*4000
height_meters = 120*4000
levels = np.arange(-60,440,20)

# Create basemap
m = Basemap(resolution='i', projection='lcc', width=width_meters,
            height=height_meters, lat_0=cen_lat, lon_0=cen_lon, lat_1=truelat1,
            lat_2=truelat2)
            
# Gridded data
x = xyz[:,0]
xi = np.linspace(x.min(), x.max(), 200)            
y = xyz[:,1]
yi = np.linspace(y.min(), y.max(), 200)
X,Y = np.meshgrid(xi, yi)
diff_grid = plt.mlab.griddata(x, y, x_diff[:,0], xi, yi)
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
p = m.contourf(lat, lon, diff_grid, cmap=plt.cm.rainbow, alpha=0.5, 
               antialiased=True, zorder=5, levels=levels)
p = m.contourf(lat, lon, diff_grid, cmap=plt.cm.rainbow, alpha=0.5, 
               antialiased=True, zorder=5, levels=levels)
obs_x, obs_y = m(obs_xy[0,:], obs_xy[1,:])
m.scatter(obs_x, obs_y, s=30, marker='^', facecolors='white', edgecolors='black', zorder=7)
cbar = m.colorbar(p, location='right', pad="5%")
cbar.set_alpha(1)
cbar.draw_all()
cbar.set_label("mm/winter")
plt.show()

#%%============================================================================
# Todos and fixmes
#==============================================================================



