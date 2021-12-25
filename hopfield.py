from PIL import Image
from os import listdir
import numpy as np

def bipolarize(img, size, array):
    for x in range(size):
        for y in range(size):
            if type(img[x, y]) == int:
                elem = img[x, y]
            else:
                elem = sum(img[x, y])/len(img[x, y])

            if elem >= 127.0:
                array = np.append(array, -1)
            else:
                array = np.append(array, 1)

    return array

## SET IMAGE SIZE AND ZERO WEIGHT MATRIX

size = 20
w = np.zeros((size, size))
examples = np.array([])

## READ IMAGES INTO BIPOLAR VECTORS

names = listdir('./examples')
for name in names:
    image = Image.open(f"./examples/{name}")

    img = image.load()
    a = np.array([])
    a = bipolarize(img, size, a)
    examples = np.append(examples, a)

    ## FILL WEIGHT FOR CURRENT VECTOR
    for x in range(size):
        for y in range(size):
            w[x, y] = w[x, y] + a[x] * a[y]

## NORMALIZE WEIGTH MATRIX
for x in range(size):
    w[x, x] = 0

for x in range(size):
    for y in range(size):
        w[x, y] = w[x, y] / size

print(w)

## RECOGNIZE TEST IMAGE
test_image = Image.open("./tests/0.png")
img = test_image.load()
test_vector = bipolarize(img, size, np.array([]))
current_vector = np.zeros(size)

answer = None
for step in range(1000):
    for x in range(size):
        d = 0
        for y in range(size):
            d += w[x, y] * test_vector[y]
            if d > 0:
                current_vector[x] = 1
            else:
                current_vector[x] = -1

    for example in examples:
        if np.array_equal(current_vector, example):
            answer = example
            break

print(answer)

