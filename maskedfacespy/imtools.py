# import the necessary packages
import cv2 
import numpy as np

def face_detector(face_net, image, threshold):
	'''
	Detects faces in the image and returns
	a list of corresponding bounding boxes
	: param face_net: OpenCV's pre-trained 
	deep neural network model
	: param image: the input image
	: param threshold: a minimum probability
	to filter out weak detections 
	'''

	# get the size of the image
	(h, w) = image.shape[:2]

	# construct a blob from the image
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
		(300, 300), (104.0, 107.0, 123.0))

	# pass the blob through the network and 
	# obtain predictions 
	face_net.setInput(blob)
	detections = face_net.forward()

	# create a container to store bounding boxes
	boxes = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence associated 
		# with the prediction 
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the 'confidence' is
		# greater than the minimum probability
		if confidence > threshold:
			# extract a bounding box and append to the list 
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

			# extract corners of the bounding box
			(startX, startY, endX, endY) = box.astype("int")

			# add the box to the list according to
			# dlib's rect format
			boxes.append((startY, endX, endY, startX))
	
	# return the list of 
	# bounding boxes
	return boxes 


def dst_pts(landmarks):
	'''
	Construct an array of destination points
	necessary to estimate the homography matrix
	'''
	# grab the coordinates of the chin
	# and nose bridge
	chin = landmarks['chin']
	nose = landmarks['nose_bridge']

	# define points
	pts1 = chin[0]
	pts2 = nose[0]
	pts3 = chin[16]
	pts4 = [chin[14][0], chin[8][1]]
	pts5 = chin[8]
	pts6 = [chin[2][0], chin[8][1]]
	
	return np.asarray([pts1, pts2, pts3, pts4, pts5, pts6])


def wear_mask(landmarks_list, image, mask_image):
	'''
	inserts the mask on detected faces based on
	the homography matrix between the input image (destination)
	and the mask image (source image)
	:param landmark_list: list of facial landmarks
	:param image: the input image 
	:param mask_image: image of the mask

	'''
	
	# grab heights and widths of 
	# the input and mask image 
	(h_i, w_i) = image.shape[:2]
	(h_m, w_m) = mask_image.shape[:2]

	# construct an array with coordinates 
	# of the points in the mask image plane
	ptsA = np.asarray([[0, 0], [w_m // 2, 0], [w_m, 0], 
		[w_m, h_m], [w_m // 2, h_m], [0, h_m]])

	# loop over the landmarks
	for landmarks in landmarks_list:
		# grab the coordinates of the chin
		#chin = landmarks['chin']

		# draw the chin
		#for c in chin:
		#	(x, y) = c 
		#	cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

		# construct an array with coordinates 
		# of the points in the input image plane
		ptsB = dst_pts(landmarks)

		# estimate the homography matrix
		(H, status) = cv2.findHomography(ptsA, ptsB)

		# warp the mask image
		mask_warped = cv2.warpPerspective(mask_image, H, (w_i, h_i), 
			flags=cv2.INTER_LINEAR)

		# convert the mask image to uint8
		mask_warped = mask_warped.astype(np.uint8)

		# grab indexes which pixel value 
		# is not zero
		imask = mask_warped > 0

		# insert the mask to the input image
		image[imask] = mask_warped[imask]

	# return the output image 
	return image.astype(np.uint8)