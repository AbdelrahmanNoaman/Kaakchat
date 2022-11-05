# KaaKChat
## Team members
*  Abdelaziz Salah 
*  Abdelrahman Noaman
*  Khaled Hesham 
*  Kirollos Samy
  
## Project Idea description:
- We propose a system that can apply real-time filters in analogy to that used in Snapchat and other social media applications. This kind of application of image processing and computer vision has a growing social and economic importance and the same techniques used can be applied in a wide variety of applications which we believe will shape the near future.
We focus mainly on applying facial filters that detect basic facial expressions and apply some real-time filters accordingly.
The first stage is to detect the presence of a face along with its location and the location of the main facial landmarks which are then fed to another stage which detects more detailed facial features and then detects the facial expressions, lastly we apply one of the filters using the data collected so far.

## functionality needed.
1. >dlib.get_frontal_face_detector()
   *   dlib.get_frontal_face_detector()	This face detector is made using the now classic Histogram of Oriented 
Gradients (HOG) feature combined with a linear classifier, an image
pyramid, and sliding window detection scheme.  This type of object detector
  is fairly general and capable of detecting many types of semi-rigid objects in addition to human faces.
(will be used directly) 

2. > dlip.shape_predictor(predictor_path)
   *    dlip.shape_predictor(predictor_path)	this function is used to detect the landmarks of the face (by passing the path of a pre trained model)

3. > isOpen()
   *    isOpen( )
	Check if the eyes or mouth is open or not. 
Ie : detect facial expressions.
(will be implemented from scratch using utility functions from opencv)

4. > Overlay()
   * This function will overlay a filter image over a face part of a person in the image/frame.
(will be implemented from scratch using utility functions from opencv)




## Project Block Diagram 
![project diagram](./projectDiagram.jpg)




## scientific papers as reference for our purposal.
* https://www.geeksforgeeks.org/ml-face-recognition-using-eigenfaces-pca-algorithm/
* https://link.springer.com/article/10.1007/s11263-018-1097-z
* https://www.eeweb.com/real-time-face-detection-and-recognition-with-svm-and-hog-features/
* https://github.com/davisking/dlib/blob/master/python_examples/face_detector.py
* https://ar.snap.com/lens-studio
  

## additional comments: 
* we believe that this project is not easy, it is tough, but we are willing to learn something new. 
* so we are hoping to can do it with the best possible scenario, but if we faced  obstacles we will need your help.