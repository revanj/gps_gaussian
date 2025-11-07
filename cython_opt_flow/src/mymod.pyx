# cython: language_level=3
cdef extern from "mylib.h":
    double square(double x)

def py_square(double x):
    return square(x)
