"""
Created on: Wed Aug  7 09:58:53 2013
Author: Nick Silverman
Email: nick.silverman11@gmail.com
Description: Driver file to principal component analysis
"""

#==============================================================================
# Import modules
#%%============================================================================
import numpy as np                                                                                                   
import matplotlib.pyplot as plt                                                
from mpl_toolkits.basemap import Basemap
from matplotlib.mlab import PCA
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
gfs_data = np.loadtxt(data_path+"win_gfs.asc")
xyz = np.loadtxt(data_path+'ascGrid.xyz', skiprows=1)[:,0:2]

#==============================================================================
# Principal Component Analysis  
#%%============================================================================
full_pca = PCA(gfs_data.T)
part_pca = PCA(gfs_data[:5,:].T)
full_y = full_pca.Y
part_y = part_pca.Y

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

# basemap
m = Basemap(resolution='i', projection='lcc', width=width_meters,
            height=height_meters, lat_0=cen_lat, lon_0=cen_lon, lat_1=truelat1,
            lat_2=truelat2)
            
# gridded data
pca = part_y[:,1]  # <--------- choose pca to plot 
x = xyz[:,0]
xi = np.linspace(x.min(), x.max(), 200)            
y = xyz[:,1]
yi = np.linspace(y.min(), y.max(), 200)
X,Y = np.meshgrid(xi, yi)
pca_grid = plt.mlab.griddata(x, y, pca, xi, yi)
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

# plot map
m.drawstates(linewidth=1, zorder=7)
m.drawcountries(linewidth=1, zorder=6)
m.imshow(elev, cmap=plt.cm.gray)
p = m.contourf(lat, lon, pca_grid, cmap=plt.cm.rainbow, alpha=0.3, 
               antialiased=True, zorder=5)
p = m.contourf(lat, lon, pca_grid, cmap=plt.cm.rainbow, alpha=0.3, 
               antialiased=True, zorder=5)
cbar = m.colorbar(p, location='right', pad="5%")
cbar.set_alpha(1)
cbar.draw_all()
cbar.set_label("")
plt.show()
#%%============================================================================
# Todos and fixmes
#==============================================================================



