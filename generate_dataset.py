# import the necessary packages
from maskedfacespy.imtools import wear_mask
from maskedfacespy.imtools import face_detector
from imutils import paths
import face_recognition
import numpy as np
import cv2
import pickle
import os 
import argparse

# create argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to dataset")
ap.add_argument("-m", "--mask", required=True,
	help="path to dataset")
ap.add_argument("-f", "--face_detector", type=str, default="hog",
	help="model for face detection: hog/cnn/dnn")
ap.add_argument("-c", "--confidence", type=float, default=0.9,
	help="minimum probability for OpenCV dnn model")
args = vars(ap.parse_args())

# define a model for face detection
model = args["face_detector"]

# load OpenCV's serialized dnn model
# if it was chosen
if model == "dnn":
	face_net = cv2.dnn.readNetFromCaffe("opencv_dnn/deploy.prototxt.txt", 
	"opencv_dnn/res10_300x300_ssd_iter_140000.caffemodel")
	threshold = args["confidence"]

# load the mask image
mask_image = cv2.imread(args["mask"])

# grab the path to images
imagePaths = list(paths.list_images(args["dataset"]))

# initialize a dictionary to 
# save bounding boxes
bbox_dict = {}

# loop over the image paths
for ind, imagePath in enumerate(imagePaths):
	# load the image
	image = cv2.imread(imagePath)

	# detect faces on the image
	# using the selected model
	if model != "dnn":
		face_locations = face_recognition.face_locations(image, model=model)
	else:
		face_locations = face_detector(face_net, image, threshold)

	# extract the facial landmarks
	face_landmarks = face_recognition.face_landmarks(image, face_locations)
	
	# wear the mask on faces if facial 
	# landmarks were extracted
	if face_landmarks:
		image = wear_mask(face_landmarks, image, mask_image)

		# show the image with the mask
		cv2.imshow("Output", image)
		key = cv2.waitKey(0) & 0xFF

		# save the image
		filename = imagePath.split('/')[-1]
		cv2.imwrite("output/{}".format(filename), image)

		# add bounding boxes to the dict
		bbox_dict[filename] = face_locations

		# if the 'q' key is pressed, 
		# stop the loop
		if key == ord("q"):
			break
		
# save the bbox dictionary data
bbox_file = open("output/bbox.pkl", "wb")
pickle.dump(bbox_dict, bbox_file)
bbox_file.close()


	
