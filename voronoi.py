import numpy as np
from math import pi, tau, sin, cos


polygons = [
    [
        (10, 10),
        (10, 15),
        (15, 15)
    ], [
        (20, 15),
        (10, 20),
        (30, 30)
    ]
]

height = 500
width  = 1000

map_2d = np.zeros((height, width), dtype="int32")

def create_map_to_3d(height, width):
    lat  = np.arange(height, dtype="float32") * (  pi / height) - pi/2
    long = np.arange(width,  dtype="float32") * (2*pi / width)

    map_3d = np.empty((height, width, 3), dtype="float32")
    x = np.cos(long) * np.cos(lat)
    y = np.sin(lat)
    z = np.sin(long) * np.cos(lat)
    map_3d[:, :, 0] = x
    map_3d[:, :, 1] = y
    map_3d[:, :, 2] = z

    return map_3d

map_to_3d = create_map_to_3d(height, width)

