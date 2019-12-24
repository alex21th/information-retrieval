#!/usr/bin/env python
"""
Simple module implementing LSH
"""

from __future__ import print_function, division
import numpy
import sys
import argparse
import time

__version__ = '0.2.1'
__author__ = 'marias@cs.upc.edu'

"""Calculates the execution time of the function next to it."""
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r %2.2f sec' %
              (method.__name__, te - ts))
        return result

    return timed

class lsh(object):
    """
    implements lsh for digits database in file 'images.npy'
    """

    def __init__(self, k, m):
        """ k is nr. of bits to hash and m is reapeats """
        # data is numpy ndarray with images
        self.data = numpy.load('images.npy')
        self.k = k
        self.m = m

        # determine length of bit representation of images
        # use conversion from natural numbers to unary code for each pixel,
        # so length of each image is imlen = pixels * maxval
        self.pixels = 64
        self.maxval = 16
        self.imlen = self.pixels * self.maxval

        # need to select k random hash functions for each repeat
        # will place these into an m x k numpy array
        numpy.random.seed(12345)
        self.hashbits = numpy.random.randint(self.imlen, size=(m, k))

        # the following stores the hashed images
        # in a python list of m dictionaries (one for each repeat)
        self.hashes = [dict() for _ in range(self.m)]

        # now, fill it out
        self.hash_all_images()

        return

    def hash_all_images(self):
        """ go through all images and store them in hash table(s) """
        # Achtung!
        # Only hashing the first 1500 images, the rest are used for testing
        for idx, im in enumerate(self.data[:1500]):
            for i in range(self.m):
                str = self.hashcode(im, i)

                # store it into the dictionary..
                # (well, the index not the whole array!)
                if str not in self.hashes[i]:
                    self.hashes[i][str] = []
                self.hashes[i][str].append(idx)
        return

    def hashcode(self, im, i):
        """ get the i'th hash code of image im (0 <= i < m)"""
        pixels = im.flatten()
        row = self.hashbits[i]
        str = ""
        for x in row:
            # get bit corresponding to x from image..
            pix = int(x) // int(self.maxval) # El pixel en el que esta el bit.
            num = x % self.maxval #Index del bit dins del pixel.
            if num <= pixels[pix]:
                str += '1'
            else:
                str += '0'
        return str

    def candidates(self, im):
        """ given image im, return matching candidates (well, the indices) """
        res = set()
        for i in range(self.m):
            code = self.hashcode(im, i)
            if code in self.hashes[i]:
                res.update(self.hashes[i][code])
        return res

    """Compares an image with images 'THAT MATCH WITH THE LSH' in the TR dataset."""
    @timeit
    def lsh_search(self, im):
        cand_set = self.candidates(im)
        minDist = numpy.inf
        index_nn = None

        for cand in cand_set:
            dist = distance(im, self.data[cand])
            if dist < minDist:
                minDist = dist
                index_nn = cand

        return (index_nn, minDist)


"""Distance function between two images."""
def distance(im1, im2):
    pixels1 = im1.flatten()
    pixels2 = im2.flatten()

    dist = 0
    for p in range(len(pixels1)):
        dist += numpy.abs(pixels1[p]-pixels2[p]) # l_1 distance!

    return dist

"""Compares an image with all images in the TR dataset."""
@timeit
def bf_search(image, medata):
    minDist = numpy.inf
    index_nn = None

    for i, tr_image in enumerate(medata):
        dist = distance(image, tr_image)
        if dist < minDist:
            minDist = dist
            index_nn = i

    return (index_nn, minDist)

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', default=20, type=int)
    parser.add_argument('-m', default=5, type=int)
    args = parser.parse_args()

    print("Running lsh.py with parameters k =", args.k, "and m =", args.m)

    # Now we calculate the nearest neighbor with brute force and with LSH.
    me = lsh(args.k, args.m)

    for r in range(1500, 1520): #1797
        im = me.data[r]
        (nn, distnn) = bf_search(im, me.data[:1500])
        (ncand, distncand) = me.lsh_search(im) # ncad pot ser NULL! Compte!!
        print(f"Image #{r}")
        print(f"The nearest neighbor with  bf_search is: {nn} with distance {distnn}")
        print(f"The nearest neighbor with lsh_search is: {ncand} with distance {distncand}")

    return


if __name__ == "__main__":
    sys.exit(main())
