import numpy as np
import cuda_example
import cv2

def read_image_rgba(filename):
    bgr_image = cv2.imread(filename)
    rgba = cv2.cvtColor(bgr_image ,cv2.COLOR_BGR2RGBA)

    return rgba

def opt_flow(first_image, second_image):
    result = np.zeros((first_image.shape[0], first_image.shape[1], 2), dtype=np.uint16)

    cuda_example.opt_flow(first_image, second_image, result)
    # result is in S10.5
    result = result.astype(np.int16).astype(np.float32) / 32.0

    return result

# first_image = read_image_rgba("002.jpg")
# second_image = read_image_rgba("003.jpg")
#
# print("type of first image is", first_image.shape, first_image.dtype)
# 
# result = np.zeros((first_image.shape[0], first_image.shape[1], 2), dtype=np.uint16)
# cuda_example.opt_flow(first_image, second_image, result)
# print(result)
