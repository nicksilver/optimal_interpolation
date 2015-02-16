"""
Created on: Wed Aug  7 09:58:53 2013
Author: Nick Silverman
Email: nick.silverman11@gmail.com
Description: Driver file to run temperature analysis
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

#==============================================================================
# Bring in data
#%%============================================================================
yrs = 9
ncells = 8639
nobs = 58
HOME = os.path.expanduser("~/")
data_path = HOME+"/SparkleShare/workspace/hist-futu_comparison/data/"
pcm_h_data = np.loadtxt(data_path+"PR_win_pcm_hist_corr.asc")
pcm_f_data = np.loadtxt(data_path+"PR_win_pcm_fut_corr.asc")
temp_data = np.loadtxt(data_path+"tempWint_fut.asc")
obs_data = np.loadtxt(data_path+"snotelPrec_matrix.txt")[:,0:9].T
obs_mask = np.loadtxt(data_path+"snotel_flag.txt")
xyz = np.loadtxt(data_path+'ascGrid.xyz', skiprows=1)[:,0:2]
dem = np.loadtxt(data_path+"dem.txt", skiprows=5)
Unc_plus = np.load("stdev_post_mean.npy")

#==============================================================================
# T-test
#%%============================================================================
mu_0 = np.zeros((pcm_h_data.shape[1]))
mu_unc = 2*Unc_plus  
sig = ttest.apply_ttest(pcm_f_data, pcm_h_data, mu=mu_0, rho=0.05, 
                        alt="two.sided")  # perform t-test actual
sig_unc = ttest.apply_ttest(pcm_f_data, pcm_h_data, mu=mu_unc, rho=0.05, 
                            alt="greater") 
unsig_unc = mu_0
unsig_unc[(sig+sig_unc)==1] = 1
xy_coord = ttest.sigcoords(xyz, sig)
xy_unc_coord = ttest.sigcoords(xyz, sig_unc)
unsig_coord = ttest.sigcoords(xyz, unsig_unc)  
xy_coord = ttest.sigcoords(xyz, sig)
diff = np.mean(pcm_f_data, axis=0) - np.mean(pcm_h_data, axis=0)  
perc_diff = (np.mean(pcm_f_data, axis=0) - np.mean(pcm_h_data, axis=0))/np.mean(pcm_h_data, axis=0)   

#==============================================================================             
# Temperature data analysis
#%%============================================================================
temp_mean = np.mean(temp_data, axis=0) - 273.15
temp = np.vstack((xyz[:,0], xyz[:,1], temp_mean, diff, unsig_unc))
temp[2,:][temp[2,:]>0] = 0
temp_cold = temp[:,np.all(temp, axis=0)].T
diff_snow = np.mean(temp_cold[:,3])  # percent difference in snowpack (below zero degrees) where change is significant

#==============================================================================
# Plot    
#%%============================================================================
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
            
sig_xy = xy_coord

# Gridded data
x = xyz[:,0]
xi = np.linspace(x.min(), x.max(), 200)  
y = xyz[:,1]
yi = np.linspace(y.min(), y.max(), 200)
X,Y = np.meshgrid(xi, yi)
temp_grid = plt.mlab.griddata(x, y, temp_mean, xi, yi)
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
v = np.arange(-12.5, 12.5, 2.5)
p = m.contourf(lat, lon, temp_grid, v, cmap=plt.cm.RdBu_r, alpha=0.3, 
               antialiased=True, zorder=5)
p = m.contourf(lat, lon, temp_grid, v, cmap=plt.cm.RdBu_r, alpha=0.3, 
               antialiased=True, zorder=5)
cbar = m.colorbar(p, location='right', pad="5%")
cbar.set_alpha(1)
cbar.draw_all()
cbar.set_label("Degrees (C)")
sig_x, sig_y = m(sig_xy[0,:], sig_xy[1,:])
m.scatter(sig_x, sig_y, s=1, marker='o', color='black', alpha=1, zorder=8)
plt.show()

#%%============================================================================
# Todos and fixmes
#==============================================================================



