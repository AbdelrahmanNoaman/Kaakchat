# Kaakchat
We have developed a face detection system that not only detects the face but also the facial landmarks with an accepted accuracy.
Our main scheme is to detect faces using segmentation according to skin color and then use an SVM model to eliminate false positives.
Next we apply corner detection to the detected face together with a best fit algorithm to detect the corners of each eye and mouth.
We applied various filters using the detected face and landmarks.

How to use?
First make sure you have installed all the required libraries on your machine, to install any library use "pip install <Library-Name>"
To run the filters: navigate to required folder and write the following command "python ./animatedSnapChatFilters.py".

Used libraries:
numpy
pandas
opencv
skimage
sklearn
mediapipe
matplotlib
pickle
random
time
os
