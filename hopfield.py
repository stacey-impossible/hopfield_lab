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

size = 4
weight = np.zeros((size*size, size*size))
names = listdir('./examples')
examples = np.zeros((len(names), size*size))

## READ IMAGES INTO BIPOLAR VECTORS

n = 0
for name in names:
    image = Image.open(f"./examples/{name}")

    img = image.load()
    img_vec = np.array([])
    img_vec = bipolarize(img, size, img_vec)
    examples[n] = img_vec
    n += 1

## FILL WEIGHTS
for x in range(size*size):
    for y in range(size*size):
        if x == y:
            weight[x, y] = 0
        else:
            weight[x, y] = np.dot(examples[:,x], examples[:,y]) / size
            weight[y, x] = weight[x, y]

## RECOGNIZE TEST IMAGE
test_image = Image.open("./tests/1.png")
img = test_image.load()
test_vect = bipolarize(img, size, np.array([]))

max_diff = size * size * 0.1

while True:
    curr_vect = np.dot(weight, test_vect)

    for x in range(size*size):
        if curr_vect[x] > 0:
            curr_vect[x] = 1
        else:
            curr_vect[x] = -1

    distance = 0
    for x in range(size*size):
        if curr_vect[x] != test_vect[x]:
            distance += 1

    print(test_vect)
    print(curr_vect)
    if distance > max_diff:
        test_vect = curr_vect
    else:
        break

print(curr_vect)

for x in range(len(examples)):
    if np.array_equal(curr_vect, examples[x]):
        print(x)

