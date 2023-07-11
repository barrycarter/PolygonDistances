#!/usr/bin/env python

'''

Boilder plate for peforming algorithm

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image, ImageFilter
import csv

# example case 2700

width = 1350
height = int(width/2)

# load raster image and note which points are lit

image = Image.open(f'usa-raster-flat-{str(width)}.png')

pixels = np.array(image)

#load raster image and note which points define the border

border = image.filter(ImageFilter.FIND_EDGES)

border_pixels = np.array(border)

#convert to interger arrays rather then boolean

pixels = pixels*1

border_pixels = border_pixels*1

#sum to defined index values for border, inside and outside. 

pixel_indicies = (pixels+border_pixels)

#We now have an 2D numpy array where:

pixel_indicies_alt = pixel_indicies - 1

#Inner pixels: 0, Outer pixels: -1, border_pixels: 1

pixel_indicies = pixel_indicies*(255/2)
print(pixel_indicies)

image = Image.fromarray(pixel_indicies)
image.show()


with open('pixels_file.txt', 'w') as f:
    csv.writer(f, delimiter=' ').writerows(pixel_indicies_alt)



