##Vehicle Detection Project


####The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/hog/hog.png
[image3]: ./examples/windows/test1.jpg.png
[image4]: ./examples/detections/test5.jpg.png
[image5]: ./examples/heatmaps/test5.jpg.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png

[image8]: ./examples/svc_vs_cnn.png

[video1]: ./project_video.mp4


###Histogram of Oriented Gradients (HOG)

####1. Loading training data

The code for this step is contained in `P5_svc_classifier_trainer.py`. I read all the `vehicle` and `non-vehicle` images. As there many small images to increase speed I have used several threads to extract features from those images.

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Because of training data have format PNG and loaded through `mpimg.imread ` it have scale from 0 to 1 instead of standard 0..255. In `src/feature_extractor.py` I check datatype of numpy array and rescale image to 0..1 if needed

####2. Extracting HOG features

The code for this step is contained in `src/feature_extractor.py`. In general was used code provided in classes, with some optimisations:

1. HOG features computed once for frame, then for every sliding window coordinates extracted from related region
2. HOG features computed only on lower part of image. 


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. HOG Parameters

I run some experiments to find parameters which gave good performance and accuracy. Finally was choosen YCrCb colorspace and HOG with 10 orientations, 8 pixels per cell and 2 cells per block, all 3 color channels was used. Problem which I encounter was that accuracy not very good indicator, as you still get many false positive detections.


####3. Training the classifier.
A LinearSVC was used as the classifier for this project. The training process can be seen in `P5_svc_classifier_trainer.py`. The features are extracted and concatenated using functions in `src/fearure_extractor.py`. 

I split data into 2 datasets: 80% for training and 20% for test.

The features include HOG features, spatial features and color histograms. The classifier is set up as a pipeline that includes a scaler as shown below:

```clf = Pipeline([('scaling', StandardScaler()),
                ('classification', LinearSVC(loss='hinge')),
               ])```

This keeps the scaling factors embedded in the model object when saved to a file. After training model saved to a picke file, and later it used in `src/svc_classifier.py` to predict class of the image. 

The model used for vehicle detection obtained a test accuracy of 0.986.

###Sliding Window Search

####1. Scales of Windows and Overlap
I create  list of windows for each scale in `src/windows_slider.py`. 
Four scales of windows were chosen - `.3, .5, .65, .8` (standard size 64x64) The larger windows closer to the driver and the smaller closer to the horizon. Overlap in both x,y was set 0.66. It could be change to balance the need for better coverage vs number of boxes generated. The more windows, the slover detection works. Windows covers only lower part of image.


![alt text][image3]

####2. Sliding window detections

Here is an example of resulting detections:

![alt text][image4]

Then from this detection heatmap is build
![alt text][image5]

## CNN Classifier

I wasn't quite happy not with speed of SVN classifier nor with accuracy for single image. My first thought was to create classifier based on Convolutional neural network which would work with the same sliding windows. With same trainign data cnn showed much better results - 1.5x speed and much less false positive detections. 

Classifier could be found at `src/cnn_classifier.py` and network was trained at `P5_cnn_trainer.py`

Difference between SVN and CNN predictions using heatmap threshhold 1
![alt text][image8]
Further research showed that there exist much faster object detection nerworks and algorithm like R-CNN, Fast R-CNN, Faster R-CNN and YOLO. 



### Video Implementation

####1. Video output)
Here's a [svc video result](./svc_project_video_vehicles.mp4) and [cnn video result](./cnn_project_video_vehicles.mp4) 


####2. Video pipeline

Working with video allowes us to use not single detection but accamulate detections from sequence of frames. Then accamulated detections could be run through the same heatmap function (check `process` in `src/vehicle_tracker.py`) with higher threshold. This method works good for cars going in the same directions, because their speed is similar to the car, but if we would need to track cars going in opposit direction this approach would fail. 


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. Main problem was false positive detections, and if use higher treshhold I get small boxes. One of solution for that could be to check size of detected label and if it too small then remove it
2. With all my optimisation I only get 1,5 frame per second of detection which is obviously not enough for real time detection. I'm interested in checking YOLO algorithm for that task
3. Difficult liting conditions make prediction harder

