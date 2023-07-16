import numpy as np
from math import sqrt, pi, tau, sin, cos, asin

import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import PIL
import time

# works just like the print function but at the end it adds the time between the present and the last call
class print_time_flag:
    t = time.time()

    def __init__(self, *args):
        new_t = time.time()
        print(*args, new_t-print_time_flag.t)
        print_time_flag.t = new_t


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

### if PIL complains, add this line ###
# PIL.Image.MAX_IMAGE_PIXELS = 536870912  # 16384 * 32768

max_width = 2048
# max_width = 32768

save_name = "ARG_US1_GB1"
save_images = True
show_images = False

images_names = [
    f"RASTER/ARG-raster-{max_width}.png",
    f"RASTER/US1-raster-{max_width}.png",
    f"RASTER/GB1-raster-{max_width}.png",
]
images_are_filled = True


def load_images(file_names):
    images = []
    for file_name in file_names:
        image = Image.open(file_name)

        images.append(np.array(image))

    return np.array(images)


full_res_images = load_images(images_names)

# todo: make this a class or something to take the functions away from their parent function
# todo: if images is a numpy array I don't need the for loop, I can do them all at once
def remove_filling(images):
    def get_expanded_image(img):
        expanded_image = np.empty((img.shape[0] + 2, img.shape[1] + 2), dtype=img.dtype)
        expanded_image[1:-1, 1:-1] = img
        expanded_image[ 0, 1:-1] = img[ 0, :]  # this should be shifted by pi, but since these are the poles
        expanded_image[-1, 1:-1] = img[-1, :]  # the horizontal distance is insignificant or even null
        expanded_image[1:-1,  0] = img[:, -1]
        expanded_image[1:-1, -1] = img[:,  0]
        # careful with corners
        return expanded_image

    def convolute(img):
        expanded_image = get_expanded_image(img)
        return img & ~(
                expanded_image[ :-2, 1:-1] &
                expanded_image[2:  , 1:-1] &
                expanded_image[1:-1,  :-2] &
                expanded_image[1:-1, 2:  ]
        )

    for i, image in enumerate(images):
        expanded_image = get_expanded_image(image)
        images[i] = convolute(image)


if images_are_filled:
    # full_res_images2 = [np.array(Image.fromarray(image).filter(ImageFilter.FIND_EDGES)) for image in full_res_images]
    # full_res_images2 = [np.array(Image.fromarray(image).filter(ImageFilter.CONTOUR)) for image in full_res_images]
    remove_filling(full_res_images)


# print("CAREFUL, TEST IMAGES")
# full_res_images = np.array([
#     np.zeros((1024, 2048), dtype="bool"),
#     np.zeros((1024, 2048), dtype="bool"),
#     # np.zeros((1024, 2048), dtype="bool")
# ])
# full_res_images[0, 500,  500] = 1
# full_res_images[1, 500, 1500] = 1
# # full_res_images[2][150, 300] = 1






# Image.fromarray(full_res_images2[0] & ~full_res_images[0]).show()
# Image.fromarray(full_res_images2[0]).show()

# full_res_images[0].show()
# exit()

def is_pow2(n):
    return n & (n - 1) == 0  # got this clever trick from https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two

def create_lower_resolution_images(full_res_images):

    # make sure that the ratio is 2x1, resolution is a power of 2, and the shape is the same for all images
    # should this check be made in the beginning of the program instead?
    # todo: if full_res_images is a np array, I need to move these checks somewhere else
    shape = full_res_images[0].shape
    h, w = shape
    assert h * 2 == w and is_pow2(h) and is_pow2(w)
    for img in full_res_images:
        assert img.shape == shape

    images = [np.copy(full_res_images)]  # careful I think this isn't clonning the arrays

    # "max pooling"
    while h > 1:
        h //= 2
        w //= 2
        print(images[-1].shape)
        images_2x = images[-1]
        images.append([])

        # todo: if full_res_images is a np array, now I can do all this at once
        for image_2x in images_2x:
            image_2x = image_2x.reshape(h, 2, w, 2)  # this allows me to split the pixels in 4
            image_1x = (
                    image_2x[:, 0, :, 0] |
                    image_2x[:, 0, :, 1] |
                    image_2x[:, 1, :, 0] |
                    image_2x[:, 1, :, 1]
            )
            images[-1].append(image_1x)
        images[-1] = np.array(images[-1])

        # Image.fromarray(images[-1][0]).show()

    return list(reversed(images))


all_res_images = create_lower_resolution_images(full_res_images)

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


# list comprehension
active_pixels = [
    ActivePixel(
        coords=np.array([0, x]),
        candidates=np.array([
            [img_i, 0, x2, 0]
            for img_i in range(len(full_res_images))
            for x2 in range(2)
            if all_res_images[0][img_i][0, x2]
        ], dtype="float32")
    )
    for x in range(2)
]

# regular for loops
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
        np.array(active_pixel_candidates, dtype="float32")
    ))
# careful that the distance value starts at 0 which isn't always true. it will need to be updated later in the code.

# for x in range(2):
#     print(active_pixels[x].candidates)
# exit()

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





# keeps track of what parts of the current res image belong to which polygon group
drawing_grid = np.array(
    [
        [-1, -1]
    ]
)


# todo: try to save distances or 3d coords if it improves performance

def split_candidate_pixels(active_pixels, images_res_x):
    for active_pixel in active_pixels:
        # print("in split for loop")
        active_pixel.candidates[:,1:3] *= 2
        new_candidates = []
        for y in range(2):
            for x in range(2):
                new_candidates.append(np.copy(active_pixel.candidates))
                new_candidates[-1][:, 1:3] += [y, x]  # todo: does the broadcasting work correctly?

        active_pixel.candidates = np.concatenate(
            new_candidates,
            axis=0
        )

        # TODO: this following filtering may make more sense to belong in a separate function??
        # remove candidate pixels that don't exist
        active_pixel.candidates = active_pixel.candidates[images_res_x[
            active_pixel.candidates[:, :3].T[0].astype("int32"),
            active_pixel.candidates[:, :3].T[1].astype("int32"),
            active_pixel.candidates[:, :3].T[2].astype("int32")],
        ]

        # print(active_pixel.candidates)

def split_active_pixels(active_pixels):
    return [
        ActivePixel(active_pixel.coords * 2 + np.array([y, x]), np.copy(active_pixel.candidates))
        for active_pixel in active_pixels
        for y in range(2)
        for x in range(2)
    ]

def incrase_drawing_grid_res(drawing_grid):
    expanded_drawing_grid = np.empty((drawing_grid.shape[0], 2,
                                      drawing_grid.shape[1], 2), dtype="int8")
    expanded_drawing_grid[:, 0, :, 0] = drawing_grid
    expanded_drawing_grid[:, 0, :, 1] = drawing_grid
    expanded_drawing_grid[:, 1, :, 0] = drawing_grid
    expanded_drawing_grid[:, 1, :, 1] = drawing_grid
    return expanded_drawing_grid.reshape(drawing_grid.shape[0] * 2,
                                         drawing_grid.shape[1] * 2)

def filter_and_conquer_active_pixels(active_pixels, res):
    res_max_error = get_res_max_error(res)

    new_active_pixels = []
    for active_pixel in active_pixels:
        if not try_conquer(active_pixel, res, res_max_error):
            new_active_pixels.append(active_pixel)

    return new_active_pixels


def get_res_max_error(res):
    pixel_distance = pi / res
    max_error = 2 * sqrt(2) * pixel_distance
    return max_error

def try_conquer(active_pixel, res, res_max_error):

    calculate_active_pixel_candidates_distance_3D(active_pixel, res)
    filter_candidates(active_pixel, res_max_error)

    if all_candidates_belong_to_the_same_polygon_group(active_pixel.candidates):
        conquer_pixel(drawing_grid, active_pixel)
        return True
    return False


def calculate_active_pixel_candidates_distance_3D(active_pixel, res):

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

    candidates_yxz = np.empty((active_pixel.candidates.shape[0], 3), dtype=active_pixel.candidates.dtype)
    candidates_yxz[:, 0] = np.sin(candidates_lat_long[:, 0])
    candidates_yxz[:, 1] = np.cos(candidates_lat_long[:, 0]) * np.sin(candidates_lat_long[:, 1])
    candidates_yxz[:, 2] = np.cos(candidates_lat_long[:, 0]) * np.cos(candidates_lat_long[:, 1])

    active_pixel.candidates[:, 3] = array_3d_distance2(candidates_yxz, active_pixel_yxz)


def array_3d_distance2(a, b):  # squared distance
    v = a - b
    return v[:, 0] * v[:, 0] + \
           v[:, 1] * v[:, 1] + \
           v[:, 2] * v[:, 2]
    # todo: maybe use @ operator for matrix multiplication?

def filter_candidates(active_pixel, res_max_error):
    # careful here, because distance is squared!
    min_distance2 = min(active_pixel.candidates[:, 3])
    threshold = get_threshold(min_distance2, res_max_error)
    # print()
    # print("previous  candidates count:", active_pixel.candidates.shape[1])
    # print(active_pixel.candidates)
    active_pixel.candidates = active_pixel.candidates[active_pixel.candidates[:, 3] <= threshold]
    # print("posterior candidates count:", active_pixel.candidates.shape[1])
    # print(active_pixel.candidates)
    # print()

def get_threshold(min_distance2, res_max_error):
    # note: distance is squared, and assumes the d column is also squared
    # it depends on how parallel the sphere surface is with respect to the line beween the two points
    min_distance = sqrt(min_distance2)
    # print("threshold", min_distance2, res_max_error, cos(asin(min_distance / 2)))
    # print("threshold", min_distance2, res_max_error, min_distance + res_max_error * cos(asin(min_distance / 2)))
    return (min_distance + res_max_error * cos(asin(min_distance / 2)))**2

def all_candidates_belong_to_the_same_polygon_group(candidates):
    return np.all(candidates[:, 0] == candidates[0, 0])

def conquer_pixel(drawing_grid, active_pixel):
    drawing_grid[tuple(active_pixel.coords)] = active_pixel.candidates[0, 0]


drawing_image = None

def draw_image(drawing_grid):
    polygon_groups_colors = np.array([
        [000, 200, 255],
        [220, 100, 120],
        [100, 100, 255],
        [000, 000, 000]

    ], dtype="uint8")

    image_grid = polygon_groups_colors[drawing_grid.reshape(-1)].reshape(*drawing_grid.shape, 3)
    drawing_image = Image.fromarray(image_grid)
    return drawing_image

def draw_polygons_over_image(drawing_image, polygons, color):
    grid = np.array(drawing_image)
    grid[polygons[0]] = grid[polygons[0]] // 2 + [127, 127, 127]
    grid[polygons[1]] = grid[polygons[1]] // 2 + [127, 127, 127]
    grid[polygons[2]] = grid[polygons[2]] // 2 + [127, 127, 127]

    return Image.fromarray(grid)


# todo: which res should it start with?
# for res, images_res_x in [(2**i, all_res_images[i]) for i in range(len(all_res_images))]:

if save_images:
    drawing_image = draw_image(drawing_grid)
    drawing_image = draw_polygons_over_image(drawing_image, all_res_images[0], color=[0, 0, 0])
    drawing_image.save(f"{save_name}_{1}x.png")


for i in range(1, len(all_res_images)):
    res = 2**i

    print("res", res)

    print_time_flag("start")

    split_candidate_pixels(active_pixels, all_res_images[i])  # todo: later try changing the order of these two and see if one is faster than the other
    print_time_flag("split_candidate_pixels")

    active_pixels = split_active_pixels(active_pixels)
    print_time_flag("split_active_pixels")

    drawing_grid  = incrase_drawing_grid_res(drawing_grid)
    print_time_flag("incrase_drawing_grid_res")

    active_pixels = filter_and_conquer_active_pixels(active_pixels, res)
    print_time_flag("filter_and_conquer_active_pixels")

    drawing_image = draw_image(drawing_grid)
    drawing_image = draw_polygons_over_image(drawing_image, all_res_images[i], color=[0, 0, 0])
    if save_images:
        drawing_image.save(f"{save_name}_{res}x.png")
    if show_images:
        drawing_image.show()











