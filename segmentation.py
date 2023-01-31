import numpy as np
import sys
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances

iterations = 5

if len(sys.argv) < 4:
    print("Error: Insufficient arguments, imageSegmentation takes three arguments")
    sys.exit()
else:
    K = int(sys.argv[1])
    if K < 3:
        print("Error: K has to be greater than 2")
        sys.exit()
    inputName = sys.argv[2]
    outputName = sys.argv[3]

# 	opening input image
image = Image.open(inputName)
imageW = image.size[0]
imageH = image.size[1]


dataVector = np.ndarray(shape=(imageW * imageH, 5), dtype=float)

pixelClusterAppartenance = np.ndarray(shape=(imageW * imageH), dtype=int)


for y in range(0, imageH):
    for x in range(0, imageW):
        xy = (x, y)
        rgb = image.getpixel(xy)
        dataVector[x + y * imageW, 0] = rgb[0]
        dataVector[x + y * imageW, 1] = rgb[1]
        dataVector[x + y * imageW, 2] = rgb[2]
        dataVector[x + y * imageW, 3] = x
        dataVector[x + y * imageW, 4] = y

# 	standardize values
dataVector_scaled = preprocessing.normalize(dataVector)

# 	set centers
minValue = np.amin(dataVector_scaled)
maxValue = np.amax(dataVector_scaled)

centers = np.ndarray(shape=(K, 5))
for index, center in enumerate(centers):
    centers[index] = np.random.uniform(minValue, maxValue, 5)

for iteration in range(iterations): 
    # 	set pixels to their cluster
    for idx, data in enumerate(dataVector_scaled):
        distanceToCenters = np.ndarray(shape=(K))
        for index, center in enumerate(centers):
            distanceToCenters[index] = euclidean_distances(
                data.reshape(1, -1), center.reshape(1, -1)
            )
        pixelClusterAppartenance[idx] = np.argmin(distanceToCenters)

    clusterToCheck = np.arange(K)  # array w/ all the clusters
    clustersEmpty = np.in1d(clusterToCheck, pixelClusterAppartenance)
    for index, item in enumerate(clustersEmpty):
        if item == False:
            pixelClusterAppartenance[
                np.random.randint(len(pixelClusterAppartenance))
            ] = index

    # 	move centers to  centroid of their cluster
    for i in range(K): 
        dataInCenter = []

        for index, item in enumerate(pixelClusterAppartenance):
            if item == i:
                dataInCenter.append(dataVector_scaled[index])
        dataInCenter = np.array(dataInCenter)
        centers[i] = np.mean(dataInCenter, axis=0)

    print("Centers Iteration num", iteration, ": \n", centers)

# 	set the pixels from the original image to be those of the pixel's cluster's centroid
for index, item in enumerate(pixelClusterAppartenance):
    dataVector[index][0] = int(round(centers[item][0] * 255))
    dataVector[index][1] = int(round(centers[item][1] * 255))
    dataVector[index][2] = int(round(centers[item][2] * 255))

# 	save the image :)
image = Image.new("RGB", (imageW, imageH))

for y in range(imageH): 
    for x in range(imageW): 
        image.putpixel(
            (x, y),
            (
                int(dataVector[y * imageW + x][0]),
                int(dataVector[y * imageW + x][1]),
                int(dataVector[y * imageW + x][2]),
            ),
        )
image.save(outputName)

##################################################
# does anybody read my code on github
# probably not
# i can say whatever i want
# just screaming into the void 
##################################################