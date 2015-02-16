'''
Created on Oct 21, 2013

@author: jared.oyler
'''
import numpy as np
from osgeo import gdal,osr
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

PROJECTION_GEO_WGS84 = 4326 #EPSG Code

class OutsideExtent(Exception):
    pass

class RasterDataset(object):

    def __init__(self,dsPath):
        
        self.gdalDs = gdal.Open(dsPath)
        
        #GDAL GeoTransform. 
        #Top left x,y are for the upper left corner of upper left pixel
        #GeoTransform[0] /* top left x */
        #GeoTransform[1] /* w-e pixel resolution */
        #GeoTransform[2] /* rotation, 0 if image is "north up" */
        #GeoTransform[3] /* top left y */
        #GeoTransform[4] /* rotation, 0 if image is "north up" */
        #GeoTransform[5] /* n-s pixel resolution */
        self.geoT = np.array(self.gdalDs.GetGeoTransform())
        
        self.projection = self.gdalDs.GetProjection()
        self.sourceSR = osr.SpatialReference()
        self.sourceSR.ImportFromWkt(self.projection)
        self.targetSR = osr.SpatialReference()
        self.targetSR.ImportFromEPSG(PROJECTION_GEO_WGS84)
        self.coordTrans_src_to_wgs84 = osr.CoordinateTransformation(self.sourceSR, self.targetSR)
        self.coordTrans_wgs84_to_src = osr.CoordinateTransformation(self.targetSR, self.sourceSR)
        
        self.min_x = self.geoT[0]
        self.max_x = self.min_x + (self.gdalDs.RasterXSize*self.geoT[1])
        self.max_y =  self.geoT[3]
        self.min_y =  self.max_y - (-self.gdalDs.RasterYSize*self.geoT[5])
        
    
    def getCoordMeshGrid(self):
        
        #Get the upper left and right point x coordinates in the raster's projection
        ulX = self.geoT[0] + (self.geoT[1] / 2.0)
        urX = self.getCoord(0, self.gdalDs.RasterXSize-1)[1]
        
        #Get the upper left and lower left y coordinates
        ulY = self.geoT[3] + (self.geoT[5] / 2.0)
        llY = self.getCoord(self.gdalDs.RasterYSize-1, 0)[0]
        
        #Build 1D arrays of x,y coords
        x = np.linspace(ulX, urX, self.gdalDs.RasterXSize)
        y = np.linspace(ulY, llY, self.gdalDs.RasterYSize)
        
        xGrid, yGrid = np.meshgrid(x, y)
        
        return yGrid, xGrid
    
    def getCoordGrid1d(self):
        
        #Get the upper left and right point x coordinates in the raster's projection
        ulX = self.geoT[0] + (self.geoT[1] / 2.0)
        urX = self.getCoord(0, self.gdalDs.RasterXSize-1)[1]
        
        #Get the upper left and lower left y coordinates
        ulY = self.geoT[3] + (self.geoT[5] / 2.0)
        llY = self.getCoord(self.gdalDs.RasterYSize-1, 0)[0]
        
        #Build 1D arrays of x,y coords
        x = np.linspace(ulX, urX, self.gdalDs.RasterXSize)
        y = np.linspace(ulY, llY, self.gdalDs.RasterYSize)
        
        return y, x
        
    def getCoord(self, row, col):
        
        xCoord = (self.geoT[0] + col*self.geoT[1] + row*self.geoT[2]) + self.geoT[1] / 2.0
        yCoord = (self.geoT[3] + col*self.geoT[4] + row*self.geoT[5]) + self.geoT[5] / 2.0
        
        return yCoord,xCoord

    def getRowCol(self,lon,lat):
        '''Returns the grid cell offset for this raster based on the input wgs84 lon/lat'''
        xGeo, yGeo, zGeo = self.coordTrans_wgs84_to_src.TransformPoint(lon,lat) 
        
        if not self.isInbounds(xGeo, yGeo):
            raise OutsideExtent("lat/lon outside raster extent: "+str(lat)+","+str(lon))
        
        originX = self.geoT[0]
        originY = self.geoT[3]
        pixelWidth = self.geoT[1]
        pixelHeight = self.geoT[5]
        
        xOffset = np.abs(np.int((xGeo - originX) / pixelWidth))
        yOffset = np.abs(np.int((yGeo - originY) / pixelHeight))
        
        row = yOffset
        col = xOffset
        
        return row,col

    def getDataValue(self,lon,lat):
        
        row,col = self.getRowCol(lon,lat)
        data_val = self.gdalDs.ReadAsArray(col,row,1,1)[0,0] 
        #data_val = self.readDataArray(col,row,1,1)[0,0]        
        return data_val

    def readAsArray(self):
        
        a = self.gdalDs.GetRasterBand(1).ReadAsArray()
        a = np.ma.masked_equal(a, self.gdalDs.GetRasterBand(1).GetNoDataValue())
        return a
    
    def isInbounds(self,x_geo,y_geo):
        return x_geo >= self.min_x and x_geo <= self.max_x and y_geo >= self.min_y and y_geo <= self.max_y

if __name__ == '__main__':
    
    #Set this to the path where you put all the data
    pathData = "/projects/daymet2/basemap_data/"
    
    dsTwxTmin = RasterDataset(pathData+'cce_topowx_tmin19812010trend.tif')
    dsDaymetTmin = RasterDataset(pathData+'cce_daymet_tmin19812010trend.tif')
    dsPrismTmin = RasterDataset(pathData+'cce_prism4km_tmin_trend1981-2010.tif')
    dsTwxTmax = RasterDataset(pathData+'cce_topowx_tmax19812010trend.tif')
    dsDaymetTmax = RasterDataset(pathData+'cce_daymet_tmax19812010trend.tif')
    dsPrismTmax = RasterDataset(pathData+'cce_prism4km_tmax_trend1981-2010.tif')
    
    #Read in trend rasters and multiply by 30 to get total trend over 30 years
    twxTmin = dsTwxTmin.readAsArray()*30
    prismTmin = dsPrismTmin.readAsArray()*30
    daymetTmin = dsDaymetTmin.readAsArray()*30
    
    twxTmax = dsTwxTmax.readAsArray()*30
    prismTmax = dsPrismTmax.readAsArray()*30
    daymetTmax = dsDaymetTmax.readAsArray()*30
    
    #Print out overall min/max trends
    print np.min([np.min(twxTmin),np.min(twxTmax),np.min(prismTmin),np.min(prismTmax),np.min(daymetTmin),np.min(daymetTmax)])
    print np.max([np.max(twxTmin),np.max(twxTmax),np.max(prismTmin),np.max(prismTmax),np.max(daymetTmin),np.max(daymetTmax)])
    
    dsGrid = dsTwxTmin
    #Get the lat,lon dimension values of the rasters
    lat,lon = dsGrid.getCoordGrid1d()
    buf = 0.25
    llcrnrlat=np.min(lat-buf)
    urcrnrlat=np.max(lat+buf)
    llcrnrlon=np.min(lon-buf)
    urcrnrlon=np.max(lon+buf)

    #Setup Basemap with a UTM Zone 12 projection
    m = Basemap(resolution='h',projection='tmerc', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,lon_0=-111,lat_0=0)
    #Convert 1-d lon/lat grid cell coordinates to a x,y mesh grid in UTM
    xMap, yMap = m(*np.meshgrid(lon,lat))
    
    #Manually setup filled contour colormap. This can be greatly simplified for other applications
    levelsRed = np.arange(0,5.5,.5)
    levelsBlue = np.arange(-3,0.5,.5)
    
    norm = Normalize(0,5)
    smReds = cm.ScalarMappable(norm, cm.Reds)
    midPtsReds = (levelsRed + np.diff(levelsRed, 1)[0]/2.0)[0:-1]
    
    norm = Normalize(-3,0)
    smBlues = cm.ScalarMappable(norm, cm.Blues_r)
    midPtsBlues = (levelsBlue + np.diff(levelsBlue, 1)[0]/2.0)[0:-1]
    
    clrsRed = [smReds.to_rgba(x) for x in midPtsReds]
    clrsBlu = [smBlues.to_rgba(x) for x in midPtsBlues]
    clrsBlu.extend(clrsRed)
    clrs = clrsBlu
    levels = np.arange(-3,5.5,.5)
    
    #Build the backgorund hillshade image
    dsElev = RasterDataset(pathData+'hillshade30-wgs84-ds.tif')
    latElev,lonElev = dsElev.getCoordGrid1d()
    #lats need to be sorted for the transform_scalar method
    latElev = np.sort(latElev)
    #get x,y resolution for the area of interest
    nx = np.sum(np.logical_and(latElev>=llcrnrlat,latElev<=urcrnrlat))
    ny = np.sum(np.logical_and(lonElev>=llcrnrlon,lonElev<=urcrnrlon))
    elev = dsElev.readAsArray()
    #flip image to match sorted lats
    elev = np.flipud(elev)
    #Reproject to UTM
    elev = m.transform_scalar(elev, lonElev, latElev, nx, ny)
    
    cfig = plt.gcf()
    #ImageGrid is a utility for setting up subplots that are more optimized for image-type data
    #Here we are setting up 6 subplots (2*3) that share a single colorbar
    grid = ImageGrid(cfig,111,nrows_ncols=(2,3),cbar_mode="single",cbar_location="right",axes_pad=0.05,cbar_pad=0.05)
   
    alpha = .5
    cbarLab = r'$^\circ$C / 30 yrs'
   
    #simple inline method for reading in CCE shapefile and drawing the boundaries on each subplot map
    def drawCceBnds(m):
        m.readshapefile(pathData+'CCE_CMP_US_Only', 'CCE_CMP_US_Only', drawbounds=True)
        m.drawcountries()
        m.drawstates()
   
    #Draw maps

    m.ax = grid[0]
    m.imshow(elev,cmap=cm.gray)
    #Bad hack: To properly draw a transparent contourf, antialiased needs to be set to True
    #and contourf needs to be called twice
    cf = m.contourf(xMap,yMap,twxTmin,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    cf = m.contourf(xMap,yMap,twxTmin,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    cbar = plt.colorbar(cf, cax = grid.cbar_axes[0])
    cbar.set_alpha(1)
    cbar.draw_all()
    grid[0].set_ylabel("Tmin")
    grid[0].set_title("TopoWx",fontsize=12)
    cbar.set_label(cbarLab)
    drawCceBnds(m)

    m.ax = grid[1]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,prismTmin,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    cf = m.contourf(xMap,yMap,prismTmin,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    grid[1].set_title("PRISM",fontsize=12)
    drawCceBnds(m)
    
    m.ax = grid[2]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,daymetTmin,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    cf = m.contourf(xMap,yMap,daymetTmin,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    grid[2].set_title("Daymet",fontsize=12)
    drawCceBnds(m)
    
    m.ax = grid[3]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,twxTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    cf = m.contourf(xMap,yMap,twxTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    grid[3].set_ylabel("Tmax")
    drawCceBnds(m)
    
    m.ax = grid[4]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,prismTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    cf = m.contourf(xMap,yMap,prismTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    drawCceBnds(m)
    
    m.ax = grid[5]
    m.imshow(elev,cmap=cm.gray)
    cf = m.contourf(xMap,yMap,daymetTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    cf = m.contourf(xMap,yMap,daymetTmax,levels=levels,colors=clrs,alpha=alpha,antialiased=True,extend="both")
    drawCceBnds(m)

    cfig.set_size_inches(11,7)
    #Uncomment and put in path to save fig
    #plt.savefig('outpath',dpi=250) 
    plt.show()