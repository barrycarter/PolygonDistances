#!/usr/bin/env python

'''

Boilder plate for peforming algorithm

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image, ImageFilter
import csv

# example case 1350

width = 1350
height = int(width/2)

# load raster image and note which points are lit

image = Image.open(f'usa-raster-flat-{str(width)}.png')

pixels = np.array(image)

#load raster image and note which points define the border

border = image.filter(ImageFilter.FIND_EDGES)

border_pixels = np.array(border)

borderPts = np.where(border_pixels != 0)

#convert to interger arrays rather then boolean

pixels = pixels*1

border_pixels = border_pixels*1

#sum to defined index values for border, inside and outside. 

pixel_indicies = (pixels+border_pixels)

#We now have an 2D numpy array where:

pixel_indicies = pixel_indicies - 1

#Inner pixels: 0, Outer pixels: -1, border_pixels: 1

#pixel_indicies = pixel_indicies*(255/2)


#Converting borderPts into a more readable array

current_border_pts = []

for i in range(len(borderPts[0])):
    current_border_pts.append([borderPts[0][i],borderPts[1][i]])


#now loop and updated current border_points:


#while -1 in pixel_indicies:

for trial in range(0,10):
    print(trial)
    for pts in current_border_pts:

        x = pts[0]
        y = pts[1]

        if x-1 > 0 and y-1 > 0 and x+1 < width and y+1 < height:

            if pixel_indicies[x-1][y] == -1:
                pixel_indicies[x-1][y] += 3

                #E
            elif pixel_indicies[x+1][y] == -1:
                pixel_indicies[x+1][y] += 3


                #S
            elif pixel_indicies[x][y-1] == -1:
                pixel_indicies[x][y-1] += 3


                #N
            elif pixel_indicies[x][y+1] == -1:
                pixel_indicies[x][y+1] += 3

                #SW    

            elif  pixel_indicies[x-1][y-1] == -1:
                pixel_indicies[x-1][y-1] += 3

                #SE
            elif pixel_indicies[x+1][y-1] == -1:
                pixel_indicies[x+1][y-1] += 3

                #NE
            elif pixel_indicies[x+1][y+1] == -1:
                pixel_indicies[x+1][y+1] += 3
        
                #NW
            elif pixel_indicies[x-1][y+1] == -1:
                pixel_indicies[x-1][y+1] += 3


    new_borderPts = np.where(pixel_indicies == 2)

    updated_border_pts = []

    for i in range(len(new_borderPts[0])):
        updated_border_pts.append([new_borderPts[0][i],new_borderPts[1][i]])

    current_border_pts = updated_border_pts


pixel_indicies_alt = (pixel_indicies + 1)*(255/3)    

image = Image.fromarray(pixel_indicies_alt)
        

image.show()

        





    