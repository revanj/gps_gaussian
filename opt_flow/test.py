import numpy as np
import cuda_example
import cv2

def read_image_rgba(filename):
    bgr_image = cv2.imread(filename)
    rgba = cv2.cvtColor(bgr_image ,cv2.COLOR_BGR2RGBA)

    return rgba

#
# # Vector operations
# a = np.random.rand(1000000).astype(np.float32)
# b = np.random.rand(1000000).astype(np.float32)
#
# c = cuda_example.vector_add(a, b)
# d = cuda_example.element_wise_multiply(a, b)
#
# # Matrix multiplication
# A = np.random.rand(512, 1024).astype(np.float32)
# B = np.random.rand(1024, 256).astype(np.float32)
#
# C = cuda_example.matrix_multiply(A, B)
#
# print(C)

#
# cuda_example.init()
first_image = read_image_rgba("002.jpg")
second_image = read_image_rgba("003.jpg")
result = np.zeros((first_image.shape[0], first_image.shape[1], 2), dtype=np.uint16)
cuda_example.opt_flow(first_image, second_image, result)
print(result)