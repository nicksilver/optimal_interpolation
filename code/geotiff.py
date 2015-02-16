"""
Created on: Mon Oct 21 19:17:52 2013
Author: Nick Silverman
Email: nick.silverman11@gmail.com

Description: This was adapted from Jared Oyler's code, see basemap_example.py.  
It is a set of general utilities to be used with geotiff and raster data. 
"""

import numpy as np
from osgeo import gdal, osr

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
        """
        Returns 2D lat lon array, similar to getCoordGrid1d
        """
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
        """
        Returns coordinates (lat/lon) of the center grid-cell
        """
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
        """
        Returns the grid cell offset for this raster based on the input wgs84 lon/lat
        """ 
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
        """
        Return data value corresponding to passed lon lat.
        """      
        row,col = self.getRowCol(lon,lat)
        data_val = self.gdalDs.ReadAsArray(col,row,1,1)[0,0] 
        #data_val = self.readDataArray(col,row,1,1)[0,0]        
        return data_val

    def readAsArray(self):
        """
        Reads geotiff as raster and masks areas with no data values.
        """        
        a = self.gdalDs.GetRasterBand(1).ReadAsArray()
        a = np.ma.masked_equal(a, self.gdalDs.GetRasterBand(1).GetNoDataValue())
        return a
    
    def isInbounds(self,x_geo,y_geo):
        return x_geo >= self.min_x and x_geo <= self.max_x and y_geo >= self.min_y and y_geo <= self.max_y
