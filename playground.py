#!/usr/bin/env python

# testing ground

from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image, ImageFilter

# from pykdtree.kdtree import KDTree

# gigapixel for the win

Image.MAX_IMAGE_PIXELS = 10**9

# the primary image (using smallest version for testing)

image = "usa-raster-flat-5400.png"

# load the primary image and turn it into an array

imMain = Image.open(image)
pixelsMain = np.array(imMain)

# compute the latitudes and longitudes of the center of each pixel in degrees

width, height = (imMain.size)

print("WH:",width,height)

# the - in lats is because the latitudes start at 90 and go down

# the (180|360)/(height|width)/2 is to get the center of the pixel

lats = -np.arange(-90, 90, 180./height)-180./height/2
lngs = np.arange(-180, 180, 360./width)+360./width/2

# convert to radians

lats = lats/180*np.pi
lngs = lngs/180*np.pi

# meshgrid so we have lng and lat for each pixel

lngs, lats = np.meshgrid(lngs, lats)

# project these points into 3D space assuming Earth is a sphere; the unit here is Earth radii, so the radius of the sphere is 1 (for now)

x = np.cos(lats)*np.cos(lngs)
y = np.cos(lats)*np.sin(lngs)
z = np.sin(lats)

# reshape to a list of points

ptsAll = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

# find the boundary points' and then their pixel coordinates

im = imMain.filter(ImageFilter.FIND_EDGES)
pixels = np.array(im)
lit = np.where(pixels != 0)

# take just the lit pixels from the 3D array we created earlier

xLit = x[lit[0],lit[1]]
yLit = y[lit[0],lit[1]]
zLit = z[lit[0],lit[1]]

# reshape the points and create a KDTree on them

pts3d = np.column_stack([xLit.flatten(), yLit.flatten(), zLit.flatten()])

# tree = cKDTree(pts3d)

print("START")

tree = cKDTree(pts3d)

# tree2 = cKDTree(ptsAll)

# since all points are within distance 2 on the unit sphere, this doesn't place a restriction, but does speed up the queries

res = tree.query(ptsAll, n_jobs=-1, distance_upper_bound=2)

print("END")

# def query_tree(point):
#    return tree.query(point, k=1)

# from concurrent.futures import ProcessPoolExecutor
# with ProcessPoolExecutor(max_workers=None) as executor:
#     res = executor.map(query_tree, ptsAll)

# res2 = tree.sparse_distance_matrix(tree2, 1)

# print(res2)

# np.savetxt('/tmp/dist.txt', res[0])

# np.save('/tmp/dist.npy', res[0])

# these are the pixels that are white in the main image

# litMain = np.where(pixelsMain != 0)

exit(0)

# these are the latitudes/longitudes of each pixel in the image

# the top left pixel is 0,0

lats = 90-180*(lit[0]+0.5)/height
# lngs = 360*(lit[0]+0.5)/width.-180.

# convert to radians


grid = 15

lats = np.linspace(-90, 90, num=grid, endpoint=True)*np.pi/180
lngs = np.linspace(-180, 180, num=grid, endpoint=True)*np.pi/180

x = np.cos(lngs)*np.cos(lats)
y = np.cos(lats)*np.sin(lngs)
z = np.sin(lats)

print("Z",z)

ptsGrid = np.column_stack([x, y, z])

print(ptsGrid.shape[0])

for i in range(np.shape(ptsGrid)[0]):
    pt = ptsGrid[i]
#    print(i, np.arcsin(pt[2])/np.pi*180, np.arctan2(pt[1], pt[0])/np.pi*180)

# 1st 180 entries are lng = -180

# 0th element is -90 south

# y=0 means lng is 0, happens at 33123 but other places far away too

# x=0 means lng is 90, happens at 

print(time.time())

dists, idxs = tree.query(ptsGrid)

print(np.shape(dists))

for i in range(dists.shape[0]):
    print(i, dists[i], ptsGrid[i])

dists = np.reshape(dists, (grid, grid))

idxs = np.reshape(idxs, (grid, grid))

print(dists)

for i in range(dists.shape[0]):
    for j in range(dists.shape[1]):
        lng = j*360/(grid-1)-180
        lat = i*180/(grid-1)-90
        dist = 8000*2*np.arcsin(dists[j][i]/2)
        idx = idxs[j][i]
        coords = pts3d[idxs[j][i]]
        latusa = np.arcsin(coords[2])
        lngusa = np.arctan2(coords[1], coords[0])
        print(lng, lat, dist, lngusa/np.pi*180, latusa/np.pi*180)

exit(0)

close = np.where(dists < 0.001)

print(close)

# min_distances = np.apply_along_axis(tree.query, 1, ptsGrid)

# print(min_distances)

print(time.time())

data = np.reshape(dists, (180, 360))

fig, ax = plt.subplots()

ax.imshow(data, cmap='rainbow')

fig.savefig("random_image.png")
