import PIL
from PIL import Image
from tkinter.filedialog import *
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from operator import add
from functools import reduce

#input the image to be compressed
file_path=askopenfilename()
imag=PIL.Image.open(file_path)

#convert image into a  nparray
img=np.array(imag)

print("IMAGE SHAPE:")
print(img.shape)

#display image
plt.imshow(img)
plt.show()

#split image into 4 subimages (one in each direction)

def split4(image):
    half_split = np.array_split(image, 2)
    res = map(lambda x: np.array_split(x, 2, axis=1), half_split)
    return reduce(add, res)

split_img = split4(img)

fig, axs = plt.subplots(2, 2)
fig.suptitle('4 subplots')

axs[0, 0].imshow(split_img[0])

axs[0, 1].imshow(split_img[1])

axs[1, 0].imshow(split_img[2])

axs[1, 1].imshow(split_img[3])
plt.show()


def concatenate4(north_west, north_east, south_west, south_east):
    top = np.concatenate((north_west, north_east), axis=1)
    bottom = np.concatenate((south_west, south_east), axis=1)
    return np.concatenate((top, bottom), axis=0)


#concatenating the subimage into the original image and displaying it
full_img = concatenate4(split_img[0], split_img[1], split_img[2], split_img[3])
plt.imshow(full_img)
plt.show()

def calculate_mean(img):
    return np.mean(img, axis=(0, 1))

means = np.array(list(map(lambda x: calculate_mean(x), split_img))).astype(int).reshape(2,2,3)

print(means)
plt.imshow(means)
plt.show()


def checkEqual(myList):
    first = myList[0]
    return all((x == first).all() for x in myList)

#Quadtree algorithm
class QuadTree:

    def insert(self, img, level=0):
        self.level = level
        self.mean = calculate_mean(img).astype(int)
        self.resolution = (img.shape[0], img.shape[1])
        self.final = True

        if not checkEqual(img):
            split_img = split4(img)

            self.final = False
            self.north_west = QuadTree().insert(split_img[0], level + 1)
            self.north_east = QuadTree().insert(split_img[1], level + 1)
            self.south_west = QuadTree().insert(split_img[2], level + 1)
            self.south_east = QuadTree().insert(split_img[3], level + 1)

        return self

    def get_image(self, level):
        if (self.final or self.level == level):
            return np.tile(self.mean, (self.resolution[0], self.resolution[1], 1))

        return concatenate4(
            self.north_west.get_image(level),
            self.north_east.get_image(level),
            self.south_west.get_image(level),
            self.south_east.get_image(level))



quadtree = QuadTree().insert(img)

plt.imshow(quadtree.get_image(1))
plt.show()
plt.imshow(quadtree.get_image(3))
plt.show()
plt.imshow(quadtree.get_image(7))
plt.show()
plt.imshow(quadtree.get_image(8))
plt.show()

print("END OF COMPRESSION.")