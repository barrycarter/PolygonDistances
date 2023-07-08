#!/usr/bin/env python

"""

Determine distances from USA to other points

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image, ImageFilter

# we start with width 1350, the smallest case 

width = 1350

# the latitudes and longitudes per pixel (height = width/2), converted
# to radians and then meshed

lats = (-90 + 180*np.arange(0.5, width/2+0.5, 1)/(width/2))/180*np.pi
lngs = (-180 + 360*np.arange(0.5, width+0.5, 1)/width)/180*np.pi

lngs, lats = np.meshgrid(lngs, lats)

# load raster image and note which points are lit

image = Image.open('usa-raster-flat-'+str(width)+'.png')

pixels = np.array(image)

print("PIXELS", pixels)

# find boundary/border points

border = image.filter(ImageFilter.FIND_EDGES)

# index the boundary points (in row, column order)

borderArray = np.array(border)

borderPts = np.where(borderArray != 0)

# select borderPts from all points at this width

x = pixels['xPts'][borderPts[0], borderPts[1]]
y = pixels['yPts'][borderPts[0], borderPts[1]]
z = pixels['zPts'][borderPts[0], borderPts[1]]

print("XYZ",x,y,z)

# reshape the points and create a KDTree on them

pts3d = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

tree = cKDTree(pts3d)

# and query against all points

ptsAll = np.column_stack([pts['xPts'].flatten(), pts['yPts'].flatten(), pts['zPts'].flatten()])

# since all points are within distance 2 on the unit sphere, this doesn't place a restriction, but does speed up the queries

res = tree.query(ptsAll, n_jobs=-1, distance_upper_bound=2)

# reshape distances to original array

print("SHAPE", np.shape(res[0]))

dist = np.reshape(res[0], [2**(zoom+8), 2**(zoom+8)])

# to "sign" the distances, we negate all distances inside original image

mainImage = np.array(image)
internal = np.where(mainImage != 0)

dist[internal] = -dist[internal]

print(dist)

print(np.percentile(dist, [0, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100]))

# rescale 0 to 1

scaled = np.round(256*(dist-np.min(dist))/(np.max(dist)-np.min(dist)))

def intToArray(x): return [x,x,x]

newarr = np.empty((len(scaled), len(scaled), 3))

for i in range(len(scaled)):
 for j in range(len(scaled[i])):
   newarr[i][j] = (scaled[i][j], scaled[i][j], scaled[i][j])

# print("SCALED", scaled, np.shape(scaled))

print("NEWARR", newarr, np.shape(newarr))

# img = Image.fromarray(newarr, mode='L')

img = Image.fromarray(np.array(newarr.astype(np.uint8)))

img.save("grayscale.png")

# print("IMG", img)

print("SCALED", scaled)

print("DIST", dist)

# plot

fig, ax = plt.subplots()

cmap = plt.get_cmap('rainbow')

plt.imshow(dist, cmap=cmap)

fig.savefig("random_image.png")

# res[0] = Euclidean distances in 3D space, res[1] = closest point index

print(np.shape(res[0]))

print("MAX", np.max(res[0]))
print("MIN", np.min(res[0]))


exit(0)

# as debugging step, save image

border.save("/tmp/usamerc.png")



print(border)

print(image)

# print(pts['xPts'][255])

print(np.shape(pts))



