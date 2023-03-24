#!/usr/bin/env python

"""

My `/usr/bin/env python --version` is `Python 2.7.5`

FAILED: could not upgrade to Python 3, my pip is broken

TODO: put github link below

To use this program, you'll need usa.png which can be found in the Images subdirectory of this git. This image is 18000x9000 so that 1 degree of latitude or longitude maps to 50 pixels. It was created as follows and (should) includes territories and possessions:

gdal_rasterize -burn 255 -where "sr_adm0_a3 = 'USA'" -ts 18000 9000 -ot Byte ne_10m_admin_0_scale_rank_minor_islands.shp -of bmp -te -180 -90 180 90 usa-raster.bmp

convert usa-raster.bmp usa-raster.png

usa-raster-small.png was created as above but with "-ts 1800 900"

usa-raster-tiny.png was created as above but with "-ts 180 90"

usa-raster-large.png was created as above but with "-ts 43200 21600"; this is the final target resolution

Note that ne_10m_admin_0_scale_rank_minor_islands.shp is a shapefile from https://www.naturalearthdata.com/ but you won't need it to run this program

TODO: use a higher resolution scale file like GADM, but note that GADM breaks the US into multiple "countries"

TODO: since gdal_rasterize is itself Python, maybe incorporate it directly instead of calling gdal_rasterize separately

Because some of the computation steps take time, the program pickles (or otherwise stores) results as it goes along, and checks to see if pickled versions of the computations already exist

"""

import os
import numpy as np
import pickle
from PIL import Image, ImageFilter

# thin wrapper around PIL.ImageFilter to find just boundary points
# given image file, after checking if it already exists; the argument
# is the filename without the png extension, not the actual filename

def compute_boundary(filename):

    outfile = "/tmp/"+filename+"-border.png"

    if os.path.exists(outfile): return

    # PIL ImageFilter does much better job of finding edges than Canny, etc
    Image.open(filename+".png").filter(ImageFilter.FIND_EDGES).save(outfile)

# compute 3D coordinates of gridded longitude and latitude matched to input resolution; pixPerDegree is the number of pixels per degree, fixed at 50 in the original version

# WARNING: with default values, this creates an 11G file in /tmp/

def lngLat2grid(filename, pixPerDegree):

    # the 3d grid doesn't actually depend on the file, but...
    outfile = "/tmp/"+filename+"-3d-grid.txt"

    if os.path.exists(outfile): return

    lats = np.linspace(-90+1/pixPerDegree/2, 90-1/pixPerDegree/2, 180*pixPerDegree, endpoint=True)*np.pi/180
    lngs = np.linspace(-180+1/pixPerDegree/2, 180-1/pixPerDegree/2, 360*pixPerDegree, endpoint=True)*np.pi/180

    x, y = np.meshgrid(lngs, lats)

    sx = np.cos(y)*np.cos(x)
    sy = np.cos(y)*np.sin(x)
    sz = np.sin(y)

    pts3d = (np.column_stack([sx.flatten(), sy.flatten(), sz.flatten()]))

    with open(outfile, 'w') as f:
        pickle.dump(pts3d, f)

# TODO: shouldn't need to mention filename more than once but ok for now

compute_boundary("usa")

print("TESTING WITH lower numbers!")

lngLat2grid("usa", 10)

with open("/tmp/usa-3d-grid.txt", "r") as f:
    pts3d = pickle.load(f)

print(np.shape(pts3d))

x = pts3d[:,0]
y = pts3d[:,1]
z = pts3d[:,2]

test = np.array([1,0,0])

print "START"
print(np.min(np.linalg.norm(pts3d-test, axis=1)))
print "END"

# print(np.linalg.norm(pts3d - test, axis=1))

# print("Starting struct creation")

# tree = KDTree(pts3d)

# tri = Triangulation(x,y,z)

# print("Done with struct creation")

# nearest_index = tri.find_simplex(test)

# nearest_point = pts3d[nearest_index]

# distance = ((point[0]-nearest_point[0])**2 + (point[1]-nearest_point[1])**2)**0.5

# print(nearest_index, nearest_point, distance)



