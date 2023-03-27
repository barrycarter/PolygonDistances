#!/usr/bin/env python

import numpy as np

lats = -np.arange(-90, 90, 180./1800)-180./1800/2
lngs = np.arange(-180, 180, 360./900)+360./900/2

# convert to radians

lats = lats/180*np.pi
lngs = lngs/180*np.pi

# meshgrid so we have lng and lat for each pixel

lngs, lats = np.meshgrid(lngs, lats)

# project these points into 3D space assuming Earth is a sphere; the unit here is Earth radii, so the radius of the sphere is 1 (for now)


for i in lats:
    for j in lngs:
        x = np.cos(i)*np.cos(j)
        y = np.cos(i)*np.sin(j)
        z = np.sin(i)


