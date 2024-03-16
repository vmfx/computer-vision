import numpy as np
from PIL import Image


def convert_to_bw(image):
    bw_image = np.array(image.convert("L"))
    return bw_image


def binarize(image, threshold=128):
    binary_mask = np.zeros_like(image, dtype=np.uint8)
    binary_mask[image < threshold] = 255 
    return binary_mask

def cutout_object(original_image, binary_mask):
    cutout_image = np.zeros_like(original_image)
    cutout_image[binary_mask == 255] = original_image[binary_mask == 255]
    return cutout_image

def process_image(my_input_image, border_value):
    input_image = Image.open(my_input_image)
    bw_image = convert_to_bw(input_image)
    binary_mask = binarize(bw_image, border_value)
    cutout_image = cutout_object(np.array(input_image), binary_mask)

    Image.fromarray(bw_image).show()
    Image.fromarray(binary_mask).show()
    Image.fromarray(cutout_image).show()

my_input_image = 'name.jpg'
border_value = 230

process_image(my_input_image, border_value)

