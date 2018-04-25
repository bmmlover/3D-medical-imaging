#!/usr/bin/python3

import numpy as np
import tifffile
import time
import gc
from scipy.ndimage import generic_filter


def timer(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r: %2.2f seconds' % (method.__name__, (te - ts)))
        return result
    return timed


def gabor_spherical_func(x, y, z, sigma, freq):
    cos = np.cos(2*np.pi*freq*((x**2 + y**2 + z**2)**0.5))
    exp = np.exp(-(x**2 + y**2 + z**2) / (2.0 * sigma**2))
    exp = exp / ((2*np.pi)**1.5 * sigma**3)
    cos = (cos - cos.min()) / (cos.max() - cos.min())
    ex = (exp - exp.min()) / (exp.max() - exp.min())
    return cos*exp


def gabor_spherical_kernel3d(sigma, freq):
    x = np.arange(-3*sigma, 3*sigma + 1)
    X, Y, Z = np.meshgrid(x, x, x)
    return gabor_spherical_func(X, Y, Z, sigma, freq)


def conv_step(arr, kernel):
    return np.sum(arr * kernel) / np.sum(kernel)


@timer
def filter3d(img, kernel):
    kwargs = dict(kernel=kernel.ravel())
    return generic_filter(img,
                          conv_step,
                          size=kernel.shape,
                          extra_keywords=kwargs)


def save_tiff(filename, array, sl=None):
    data = array / np.max(array)
    if sl is not None:
        data = data[sl, sl]
    data = 255 * np.transpose(data, (2, 0, 1))
    converted = data.astype(np.uint8)
    tifffile.imsave(filename + '.tif', converted)
    print(filename + '.tif saved!')


stack = tifffile.imread('b08015.tif')
stack = np.transpose(stack, (1, 2, 0))
print(stack.shape)
gc.collect()

#for i in np.arange(0.5, 2, 0.2):
for i in [0.5, 0.7, 0.9, 1, 1.1, 1.3, 1.5, 1.7, 1.9]:
    gab = gabor_spherical_kernel3d(i, 1/i)
    gstack = filter3d(stack, gab)
    save_tiff('bgsph' + str(i), gstack)
    del gab
    del gstack
    gc.collect()
