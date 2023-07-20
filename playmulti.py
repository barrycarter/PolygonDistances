# testbed

# global imports

import geopandas as gpd
import fiona
import shapely
import pickle
import json
import random
import numpy as np
import geopandas as gpd
from scipy.sparse import csr_matrix
from scipy.spatial import SphericalVoronoi, cKDTree
from PIL import Image, ImageFilter

# import my lib
from bclib import *

# big pictures

Image.MAX_IMAGE_PIXELS = 10**9

# longest segment finding?

def play8():

    sf = gpd.read_file('ne_10m_admin_0_scale_rank_minor_islands.shp')
#    sf = gpd.read_file('densekm.shp')

    # count how many points
    count = 0

    for i in sf['geometry'].exterior:
#        print("I",i)

        count = count + len(i.coords)

        print("COUNT", count)

        for j in zip(i.coords[:-1], i.coords[1:]):
            pass
#            print(j)
#            print(np.cos(j[0][1]/180*np.pi)*np.abs(j[0][0]-j[1][0]))
#            print(np.abs(j[0][1]-j[1][1]))

#    print("COUNT", count)

#            pt1 = gpd.Point((j[0]))
#            print(new gpd.Point(j[0]).distance(new gpd.Point[j[1]]))



#            print("J", j[0])
        # for j in i.exterior.coords:

#            print("J", j)
#        print(debug0(object=i))
#        for seg in i['geometry']:
#            print("SEG", seg)

#    print(sf['geometry'])



# read part of a shapefile

def play1():

    # Load the shapefile (this is what takes the time, regardless of chunksize)

    shapefile = gpd.read_file("ne_10m_admin_0_scale_rank_minor_islands.shp", chunksize=100)

    # Create a select query
#    query = "sr_sov_a3 = 'US1'"

#    print(shapefile.query(query))

    # Select the features that match the query
    selected_features = shapefile[shapefile["sr_sov_a3"] == "US1"]

    debug0(object = selected_features)

    # loop

#    for i in selected_features:
#        debug0(object = i)

    # Print the selected features
#    print(selected_features)

def play2():

    sf = shapefile.Reader("ne_10m_admin_0_scale_rank_minor_islands.shp")

    shps = sf.shapes()

    print(shps)


def play3():

    arr = {}

    width = 1800
    height = width/2

    lats = (-np.arange(-90, 90, 180./height)-180./height/2)/180*np.pi
    lngs = (np.arange(-180, 180, 360./width)+360./width/2)/180*np.pi

    lngs, lats = np.meshgrid(lngs, lats)

    x = np.round(6371*np.cos(lats)*np.cos(lngs))
    y = np.round(6371*np.cos(lats)*np.sin(lngs))
    z = np.round(6371*np.sin(lats))

#    for i in x: print("X", i)
#    for i in y: print("Y", i)
#    for i in z: print("Z", i)

    ptsAll = np.column_stack([x.flatten(), y.flatten(), z.flatten()]).astype(np.int32)

    for i in ptsAll: arr[tuple(i)] = 1
#    print(arr)
    print(arr.get((107, -21, -6370)))

#    pts2 = np.unique(ptsAll, axis=0)

#    print("PTS", ptsAll)
#    print("PTS2", pts2)
#    print("PTSALL", ptsAll.shape, pts2.shape)

def play4():

    arr = {}
    arr[4,5,6] = 7

    if (4,5,6) in arr: print("YES")
    if (1,2,4) in arr:
        print("YES")
    else:
        print("NO")
    for i in arr: print("I", i)
#    print(arr)
#    try: 
#        x = arr[1,2,3]
#    finally:
#        x = 0

def widthHeight2xyz(width, height):
    """

Given a width and height, return xyz values of width points of
longitude equally spaced and height points of latitude equally spaced

    """

    lats = (-np.arange(-90, 90, 180./height)-180./height/2)/180*np.pi
    lngs = (np.arange(-180, 180, 360./width)+360./width/2)/180*np.pi

    print("LATS", lats)
    print("LNGS", lngs)

    lngs, lats = np.meshgrid(lngs, lats)

    x = np.cos(lats)*np.cos(lngs)
    y = np.cos(lats)*np.sin(lngs)
    z = np.sin(lats)

    return np.column_stack([x.flatten(), y.flatten(), z.flatten()])

def image2xyz(filename):

    """

Given an image filename, treat that image as an equiangular map and return the lit pixels in xyz format

    """

    imMain = Image.open(filename)
    width, height = imMain.size

    edge1 = imMain.filter(ImageFilter.FIND_EDGES)
#    edge1.save("/tmp/temp1.png")

    edgePix1 = np.where(np.array(edge1) != 0)
    print(np.shape(edgePix1))

    filter4 = ImageFilter.Kernel((3, 3), (0, -1, 0, -1, 4, -1, 0, -1, 0), 1, 0)

    edge2 = imMain.filter(filter4)
    edgePix2 = np.where(np.array(edge2) != 0)
    print(np.shape(edgePix2))
#    edge2.save("/tmp/temp2.png")

    return

    allPixels = np.array(imMain)
    litPixels = np.where(allPixels != 0)
    print(litPixels)
    litPixels = litPixels[0]*width + litPixels[1]
    print(litPixels)
    print(width, height)

def play5():

    print(random.uniform(0,1))
    return
    points = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0, -1, 0], [-1, 0, 0]])
    center = np.array([0, 0, 0])
    radius = 1
    sv = SphericalVoronoi(points, radius, center)
    debug0(object=sv)
    
# store edge pixels to JSON

def play6(filename):

    imMain = Image.open(filename)
    width, height = imMain.size
    edges = imMain.filter(ImageFilter.Kernel((3, 3), (0, -1, 0, -1, 4, -1, 0, -1, 0), 1, 0))
    edgePix = np.where(np.array(edges) != 0)
    pixNums = edgePix[0]*width + edgePix[1]
    edges.save("/tmp/temp3.png")
    print(pixNums)
    np.savetxt("/tmp/temp4.txt", pixNums, fmt = '%d', newline=',')
#    with open('/tmp/arr.json', 'w') as f:
#        json.dump(pixNums, f, indent=4)

def raster2JSONPixels(**obj):

    """

Converts a raster map into a .js file defining a variable with the height, weight, and edge pixels. Input:

raster: the filename containing the raster map
var: the variable to assign in the js
outfile: the name of the output file

    """

    imMain = Image.open(obj['raster'])
    width, height = imMain.size

    edges = imMain.filter(ImageFilter.Kernel((3, 3), (0, -1, 0, -1, 4, -1, 0, -1, 0), 1, 0))

    edgePix = np.where(np.array(edges) != 0)

    pixNums = edgePix[0]*width + edgePix[1]

#     str = obj[var] + "= {"height": " + height +, "width": {width}, "points": ['
    # the double { and } below are to escape them, JS vs Python
    str = f'{obj["var"]} = {{"height": {height}, "width": {width}, "points": '

    str += np.array2string(pixNums, separator=', ', max_line_width = np.inf, threshold = np.inf)

    str += "};"

    with open(obj['outfile'], 'w', encoding='utf-8') as f:
        f.write(str)

def raster2KDTree(**obj):

    """

Given a raster image presumed to be an equiangular map, return a 3D KDTree of the edge points projected to 3 dimensions. Input:

raster: the filename containing the raster map

    """

    imMain = Image.open(obj['raster'])
    width, height = imMain.size

    # the latitudes and longitudes of pixel centers in radians

    lats = (-np.arange(-90, 90, 180./height)-180./height/2.)*np.pi/180.
    lngs = (np.arange(-180, 180, 360./width)+360./width/2.)*np.pi/180.

    edges = imMain.filter(ImageFilter.Kernel((3, 3), (0, -1, 0, -1, 4, -1, 0, -1, 0), 1, 0))

    edgePix = np.where(np.array(edges) != 0)

    # the latitudes and longitudes for the edge points

    lats = lats[edgePix[0]]
    lngs = lngs[edgePix[1]]

    x = np.cos(lats)*np.cos(lngs)
    y = np.cos(lats)*np.sin(lngs)
    z = np.sin(lats)

    # reshape to array of 3D points

    ptsAll = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

    return cKDTree(ptsAll)

def slippyTile2Closest(**obj):

    """

Given a slippy tile in xyz format and list of CKDTrees, determine which of the CDKTrees (points) are closest to the center pixel of the slippy tile. Input:

x, y, z: slippy tile coordinates
trees: a list of CKDTrees

TODO: check to see if the min applies to the whole tile

    """

    latLng = tile2LngLat(z=obj['z'], x=obj['x']+0.5, y=obj['y']+0.5)
    lat = latLng['lat']
    lng = latLng['lng']

    # this is the angle length across the whole tile and is therefore MUCH larger than the minimum value here which would be the half diagonal length as a straight line

    delta = 2*np.pi/2**obj['z']
    print("DELTA: ",delta)

    x = np.cos(lat)*np.cos(lng)
    y = np.cos(lat)*np.sin(lng)
    z = np.sin(lat)

    dists = []

    for i in trees:
        dists.append((i.query([x,y,z], k=1)[0]))

    print("DISTS", dists)
    print(np.argsort(dists))

def tile2LngLat(**obj):

    """

Ported from bclib.js.

Converts a slippy tile to a latitude and longitude value:

Input object:

z: tile zoom value
x: tile x value (may be fractional, 0 = left edge of tile, 0.5 = middle of leftmost pixel)
y: tile y value (may be fractional, 0 = top edge of tile, 0.5 = middle of top pixel of tile)

Output object:

lng: longitude in radians corresponding to z/x/y
lat: latitude in radians corresponding to z/x/y

    """

    ret = {};

    # the longitude is linear
    ret['lng'] = obj['x'] / 2 ** obj['z'] * 2 * np.pi - np.pi;

    # for Mercator (http://mathworld.wolfram.com/MercatorProjection.html)

    ret['lat'] = np.pi / 2 - 2 * atan(np.exp((obj['y'] / 2 ** obj['z'] - 1 / 2) * 2 * np.pi))

    return ret;


def play9():

    # Set the filepath to the shapefile
    filepath = "densekm.shp"

    # Read the shapefile into a GeoDataFrame
    df = gpd.read_file(filepath)

    # Create a SQL query
    sql = "sr_sov_a3 == 'GHA'"

    # Select the features from the GeoDataFrame that match the SQL query
    df = df.query(sql)

    # Print the GeoDataFrame

    for i in df['geometry'].exterior:
        print(list(i.coords))


######### TESTING BELOW THIS LINE, NO MORE FUNCTIONS PLEASE ########

play9()

exit()

play8()

exit()

with open("US1.tree", "rb") as f:
    tree = pickle.load(f)

print(tree.query([1,0,0], 1))
print(tree.query([0,1,0], 1))
print(tree.query([0,0,1], 1))

exit()

for i in ['US1', 'GB1', 'ARG']:
    with open(f'{i}.tree', 'wb') as f:
        pickle.dump(raster2KDTree(raster = f'RASTER/{i}-raster-43200.png'), f)


exit()

tree = raster2KDTree(raster = "RASTER/US1-raster-43200.png")

with open('/tmp/tree.txt', 'wb') as f:
    pickle.dump(tree, f)

exit()

trees = [raster2KDTree(raster = "RASTER/US1-raster-43200.png"),
         raster2KDTree(raster = "RASTER/GB1-raster-43200.png"),
         raster2KDTree(raster = "RASTER/ARG-raster-43200.png")
         ]

slippyTile2Closest(z=11, x=417, y=810, trees = trees)
slippyTile2Closest(z=6, x=21, y=39, trees = trees)
slippyTile2Closest(z=8, x=82, y=171, trees = trees)
slippyTile2Closest(z=9, x=164, y=342, trees = trees)
slippyTile2Closest(z=0, x=0, y=0, trees = trees)
slippyTile2Closest(z=9, x=165, y=229, trees = trees)

# print(trees)

# play6("usa-raster-flat-1350.png")

# raster2JSONPixels(raster="RASTER/US1-raster-43200.png", var="usa", outfile="usa.js")

# raster2JSONPixels(raster="RASTER/GB1-raster-43200.png", var="uk", outfile="uk.js")

# raster2JSONPixels(raster="RASTER/ARG-raster-43200.png", var="arg", outfile="arg.js")

exit()

# play5()

# pts = widthHeight2xyz(20, 10)

# print(pts)

# print(np.shape(pts))

# print(image2xyz("usa-raster-flat-1350.png"))

# print(image2xyz("playground.png"))

# print(image2xyz("usa-raster-flat-43200.png"))



