# import the necessary packages
import numpy as np 
import argparse 
import cv2 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image 
image = cv2.imread(args["image"])

lower = [50, 50, 50]
upper = [255, 255, 255]

lower = np.array(lower, dtype = "uint8")
upper = np.array(upper, dtype = "uint8")

# find the colors within the specified boundaries and apply
# the mask 
mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask=mask)

# show the images
cv2.imshow("Output", np.hstack([image, output]))
cv2.waitKey(0)