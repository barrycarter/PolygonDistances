import numpy as np
from math import sqrt, pi, tau, sin, cos, asin

import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

# error margin within distances in candidate pixels:
#   distance could be added or substracted by sqrt(2)
#   so distance offset should be 2*sqrt(2)
#   this would be multiplied by the distance between a pixel and an adjacent one (in 3d)
#   it's possible this window will be too big for very long distances,
#   since in the sphere, moving one pixel will only equate to a very small distance.
#   so, to find a more accurate error margin...
#   when choosing the filter gap, consider the distance between the active pixel and the closest pixel,
#    and consider what's the maximum distance that could be off.
#    for example, if the arc length is pi/2, then the max possible error would be multiplied by cos(pi/4) or something like that.
#    probably you need a tiiiiiiiiny bit of extra margen, because this calculation itself may also be affected by an error.
#    maybe this is the (a) formula:  # may be not the theoretical best
#     error_margin = 2 * sqrt(2) *  .....

images_names = [
    "RASTER/ARG-raster-2048.png",
    "RASTER/US1-raster-2048.png",
    "RASTER/GB1-raster-2048.png",
]
images_are_filled = True


def load_images(file_names):
    images = []
    for file_name in file_names:
        image = Image.open(file_name)

        images.append(np.array(image))
        # images.append(image)

    return images


full_res_images = load_images(images_names)


def remove_filling(images):
    def get_expanded_image(img):
        expanded_image = np.empty((img.shape[0] + 2, img.shape[1] + 2), dtype=img.dtype)
        expanded_image[1:-1, 1:-1] = img
        expanded_image[0, 1:-1] = img[0, :]  # this should be shifted by pi, but since these are the poles
        expanded_image[-1, 1:-1] = img[-1, :]  # the horizontal distance is insignificant or even null
        expanded_image[1:-1, 0] = img[:, -1]
        expanded_image[1:-1, -1] = img[:, 0]
        # careful with corners
        return expanded_image

    def convolute(img):
        expanded_image = get_expanded_image(img)
        return img & ~(
                expanded_image[:-2, 1:-1] &
                expanded_image[2:, 1:-1] &
                expanded_image[1:-1, :-2] &
                expanded_image[1:-1, 2:]
        )

    for i, image in enumerate(images):
        expanded_image = get_expanded_image(image)
        images[i] = convolute(image)


if images_are_filled:
    # full_res_images2 = [np.array(Image.fromarray(image).filter(ImageFilter.FIND_EDGES)) for image in full_res_images]
    # full_res_images2 = [np.array(Image.fromarray(image).filter(ImageFilter.CONTOUR)) for image in full_res_images]
    remove_filling(full_res_images)


# Image.fromarray(full_res_images2[0] & ~full_res_images[0]).show()
# Image.fromarray(full_res_images2[0]).show()

# full_res_images[0].show()
# exit()


def create_lower_resolution_images(full_res_images):
    def is_pow2(n):
        return n & (
                    n - 1) == 0  # got this clever trick from https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two

    # make sure that the ratio is 2x1, resolution is a power of 2, and the shape is the same for all images
    # should this check be made in the beginning of the program instead?
    shape = full_res_images[0].shape
    h, w = shape
    assert h * 2 == w and is_pow2(h) and is_pow2(w)
    for img in full_res_images:
        assert img.shape == shape

    images = [[*full_res_images]]  # careful I think this isn't clonning the arrays

    # "max pooling"
    while h > 1:
        h //= 2
        w //= 2
        images_2x = images[-1]
        images.append([])

        for image_2x in images_2x:
            image_2x = image_2x.reshape(h, 2, w, 2)  # this allows me to split the pixels in 4
            image_1x = (
                    image_2x[:, 0, :, 0] |
                    image_2x[:, 0, :, 1] |
                    image_2x[:, 1, :, 0] |
                    image_2x[:, 1, :, 1]
            )
            images[-1].append(image_1x)

        # Image.fromarray(images[-1][0]).show()

    return list(reversed(images))


all_res_images = create_lower_resolution_images(full_res_images)


# # todo: check if this function is working right
# # TODO: is this function necessary? do I need to calculate every single pixel? I think this old approach isn't convenient anymore
# #         perhaps I'll need some storing variable tho, in order to prevent recalculating values
# def create_map_3d(height, width):
#     lat  = (np.arange(height, dtype="float32") * (  pi / height) - pi/2 ).reshape(-1,  1)
#     long = (np.arange(width,  dtype="float32") * (2*pi / width )        ).reshape( 1, -1)
#
#     map_3d = np.empty((height, width, 3), dtype="float32")
#     x = np.cos(long) * np.cos(lat)
#     y = np.sin(lat)  * np.ones((height, width), dtype="float32")
#     z = np.sin(long) * np.cos(lat)
#     map_3d[:, :, 0] = x
#     map_3d[:, :, 1] = y
#     map_3d[:, :, 2] = z
#
#     return map_3d
# map_3d = create_map_3d(height, width)

def array_3d_distance2(a, b):  # squared distance
    v = a - b
    return v[:, 0] * v[:, 0] + \
           v[:, 1] * v[:, 1] + \
           v[:, 2] * v[:, 2]
    # maybe use @ operator for matrix multiplication?


# active pixels contain the coordinates of themselves and an array with all the candidate "owners"
# candidates have 4 coordinates, the image index, y, x, and the last one is use to later compute the distance
#                                                       (could be changed later) so that I can filter them
# this structure could likely be changed later.

# there are two initial active pixels
# each one of them has


class ActivePixel:
    def __init__(self, coords, candidates):
        self.coords = coords
        self.candidates = candidates


active_pixels = [
    ActivePixel(
        coords=np.array([0, x]),
        candidates=np.array([
            [img_i, 0, x2, 0]
            for img_i in range(len(full_res_images))
            for x2 in range(2)
            if all_res_images[0][img_i][0, x2]
        ])
    )
    for x in range(2)
]

active_pixels = []
for x in range(2):
    active_pixel_coords = np.array([0, x])
    active_pixel_candidates = []

    for img_i in range(len(full_res_images)):
        for x2 in range(2):
            if all_res_images[0][img_i][0, x2]:
                active_pixel_candidates.append([img_i, 0, x2, 0])

    active_pixels.append(ActivePixel(
        active_pixel_coords,
        np.array(active_pixel_candidates)
    ))
# careful that the distance value starts at 0 which isn't always true. it will need to be updated later in the code.

for x in range(2):
    print(active_pixels[x][1])
exit()

# a possible optimization is to save the relative coordinates of the candidates (nvm, I need them to get the 3d ones? (I'm sleepy already))
# idea for the future, store things vertically instead of horizontally
#   I mean that you store all the first candidates of the active pixels in a single numpy array
#   then you store all the second ones for the ones that have one
#   then all the third ones
#   etc
#   each array will be shorter
#   and you make calculations in this dimension if this dimension is significantly larger
#   another idea is to sort active pixels by number of candidate pixels
#   and then group them and do them all at once, sounds good too


# maybe store all the active pixels indices in the same array
# ohhh a full numpy approach:
#   active pixels is a single numpy array
#   every row contains the data of an active pixel
#    it contains the x and y coordinates
#    and next to it all the candidates..... well, actually if I do this I'll calculate too much
#    space isn't a problem, processing is

# def get_3d_coords_from_array(arr):
#     lat  = (np.arange(height, dtype="float32") * (pi / height) - pi / 2).reshape(-1, 1)
#     long = (np.arange(width,  dtype="float32") * (2*pi / width )        ).reshape( 1, -1)
#
#     d =
#
#     map_3d = np.empty((height, width, 3), dtype="float32")
#     x = np.cos(long) * np.cos(lat)
#     y = np.sin(lat)  * np.ones((height, width), dtype="float32")
#     z = np.sin(long) * np.cos(lat)
#     map_3d[:, :, 0] = x
#     map_3d[:, :, 1] = y
#     map_3d[:, :, 2] = z
#
#     return map_3d

# keeps track of what parts of the current res image belong to which polygon group
drawing_grid = np.array(
    [
        [-1, -1]
    ]
)


def calculate_active_pixel_candidates_distance_3D(active_pixel, res):
    lat  = pi * active_pixel.coords[0] / res - pi / 2
    long = pi * active_pixel.coords[1] / res - pi  # it's not 2 pi, it get's cancelled with x
    shift = np.array([-pi / 2, -pi])

    # 2D:
    active_pixel_lat_long = (pi / res) * active_pixel.coords + shift
    candidates_lat_long   = (pi / res) * active_pixel.candidates[:, 1:3] + shift.reshape(1, 2)

    # 3D:
    active_pixel_yxz = np.array([
        sin(active_pixel_lat_long[0]),
        cos(active_pixel_lat_long[0]) * sin(active_pixel_lat_long[1]),
        cos(active_pixel_lat_long[0]) * cos(active_pixel_lat_long[1])
    ])
    candidates_yxz = None
    # TODO: INCOMPLETE


def get_threshold(min_distance, res_max_error):
    # it depends on how parallel the sphere surface is with respect to the line beween the two points
    return min_distance + res_max_error * cos(asin(min_distance / 2))
    # perhaps should do "+ cos(asin(min_distance+()/2))"


def filter_candidates(active_pixel, res_max_error):
    min_distance = min(active_pixel.candidates[:, 3])
    threshold = get_threshold(min_distance, res_max_error)
    active_pixel.candidates = active_pixel.candidates[active_pixel.candidates[:, 3] <= threshold]


def all_candidates_belong_to_the_same_polygon_group(candidates):
    return np.all(candidates[:, 0] == candidates[0, 0])


def get_res_max_error(res):
    pixel_distance = pi / res
    max_error = 2 * sqrt(2) * pixel_distance
    return max_error


def conquer_pixel(drawing_grid, active_pixel):
    drawing_grid[tuple(active_pixel.coords)] = active_pixel.candidates[0, 0]


def try_conquer(active_pixel, res, res_max_error):

    calculate_active_pixel_candidates_distance_3D(active_pixel, res)
    filter_candidates(active_pixel, res_max_error)

    if all_candidates_belong_to_the_same_polygon_group(active_pixel.candidates):
        conquer_pixel(drawing_grid, active_pixel)
        return True
    return False


def filter_and_conquer_active_pixels(active_pixels, res):
    res_max_error = get_res_max_error(res)

    return [
        active_pixel
        for active_pixel in active_pixels
        if not try_conquer(active_pixel, res, res_max_error)
    ]

def split_active_pixels(active_pixels):
    return [
        ActivePixel(active_pixel.coords * 2 + np.array([y, x]), np.copy(active_pixel.candidates))
        for active_pixel in active_pixels
        for y in range(2)
        for x in range(2)
    ]

def split_candidate_pixels(active_pixels):
    for active_pixel in active_pixels:
        active_pixel.candidates[:,1:3] *= 2
        new_candidates = []
        for y in range(2):
            for x in range(2):
                new_candidates.append(np.copy(active_pixel.candidates))
                new_candidates[-1][1:3] += np.array([y, x])  # todo: does the broadcasting work correctly?

        active_pixel.candidates = np.concatenate(
            new_candidates,
            axis=0
        )

def incrase_drawing_grid_res(drawing_grid):
    expanded_drawing_grid = np.empty((drawing_grid.shape[0], 2,
                                      drawing_grid.shape[1], 2), dtype="int8")
    expanded_drawing_grid[:, 0, :, 0] = drawing_grid
    expanded_drawing_grid[:, 0, :, 1] = drawing_grid
    expanded_drawing_grid[:, 1, :, 0] = drawing_grid
    expanded_drawing_grid[:, 1, :, 1] = drawing_grid
    return expanded_drawing_grid.reshape(drawing_grid.shape[0] * 2,
                                         drawing_grid.shape[1] * 2)

# increase resolution by 2
for res in (2 ** i for i in range(len(all_res_images))):
    split_candidate_pixels(active_pixels)  # todo: later try changing the order of these two and see if one is faster than the other
    active_pixels = split_active_pixels(active_pixels)
    drawing_grid  = incrase_drawing_grid_res(drawing_grid)

    active_pixels = filter_and_conquer_active_pixels(active_pixels, res)




