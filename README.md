# masked-face
COVID-19 pandemic has changed our social life. We need to keep social distancing and wear a mask in public places. The latter is a problem for existing face detection/recognition deep learning models which were trained on faces without masks. The solution might be to train these networks on a dataset which contains masked faces. One of the possible options to create the mentioned dataset is to collect images from the Internet and label them manually. It takes large amount of time and human effort. Another option is to artificially "wear" the mask on faces.

This repo allows us to automatically detect faces on the image using HOG/CNN models and extract facial landmarks of each face. Then, it uses facial landmarks to estimate a homography matrix between the face and an image of the mask. Finally, the image of the mask is "worn" to the face using the homography matrix. 

## Dependencies
1. Ubuntu 16.04
2. Python 3.5.2 
3. OpenCV 4.0.0
4. NumPy
5. imutils
6. face_recognition https://github.com/ageitgey/face_recognition 

## How to install packages?
pip3 install imutils 
pip3 install face_recognition

## How to use?
1. Clone this github repo
*git clone https://github.com/akuzdeuov/masked-face.git
*cd masked-face

2. The main script **generate_dataset.py** has the following input arguments:
**--dataset**: path to the input dataset with images.
**--mask**: image of the mask. In case if you want to use another image of the mask then make sure that its background is absolutely black.
**--face_detector**: model for face detection. You have three options: **hog/cnn/dnn**. **hog** - histogram of oriented gradients algorithm and **cnn** - convolutional neural network based model. Both models are provided by **face_recognition** library. The last option **dnn** is a pre-trained deep neural network model which comes with OpenCV 4.0.0. The default option is **hog**.
**--confidence**: a minimum probability for OpenCV's **dnn** model. The default value is 0.9 (90%).

3. Run the code:
**python generate_dataset.py --dataset dataset/ --mask blue_mask.png** 

4. Output images and a pickle file with bounding box coordinates are saved in **./output** folder.

## Example outputs
### Input:
![example2_r](https://github.com/akuzdeuov/masked-face/blob/master/dataset/example_2.jpg)
### Ouput:
![example2](https://github.com/akuzdeuov/masked-face/blob/master/output/example_2.jpg)

### Input:
![example1_r](https://github.com/akuzdeuov/masked-face/blob/master/dataset/example_1.jpg)
### Output:
![example1](https://github.com/akuzdeuov/masked-face/blob/master/output/example_1.jpg)

Here we can see that some faces were not detected on the second image. Because we use **hog** by default. If we switch to **cnn** mode which is more accurate but slower:
![example_1cnn](https://github.com/akuzdeuov/masked-face/blob/master/output/example_1_cnn.jpg)
