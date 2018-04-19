# Visual Odometry

This document will explain the theory behind visual odometry and show an implementation that can be used on a Turtlebot using Python and OpenCV.

## Visual Odometry Overview

Visual odometry allows you to determine a path traveled using only image data from a camera. In this example we will use a single calibrated camera.

The following steps will be described:

-  Feaure detection
-  Optical flow
-  Calculating the essential matrix
-  Calculating pose/updating position

## Feature Detection

The first step is to detect features in the image that can be tracked. OpenCV has several ways of doing this, but here we will use the FAST feature detector. Essentially, this algorithm searches for corners in an image by taking each pixel and looking at its surrounding pixels to see if they are significantly lighter and darker than the pixel in question. For more details of the theory behind this algorithm, click [here](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_fast/py_fast.html).

This is the code used to initialize the feature detector and then find keypoints in the current frame. The x and y pixel coordinates of each keypoint are then extracted and formatted so that they are ready to be used in the next step.

```python
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find the keypoints
kp = fast.detect(frame1,None)
# Extract the x and y coordinates from the keypoints
p0 = np.float32([p.pt for p in kp])
p0 = p0.reshape(-1,1,2)
```
In the figure below, you can see an image taken by the Turtlebot's camera with the feature points detected by the FAST algorithm.

![alt text](https://github.com/keenan573/visual-odometry/images/FASTfeatures2.png "Camera image with feature points")

## Optical Flow

The next step is to determine how much each feature we detected previously moves from one frame to the next. This is the basis for being able to determine the motion of the camera in 3D space. In OpenCV this can be done using the Lucas-Kanade method, described [here](https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html).

The following code takes the points found by the feature detector in the previous frame, and returns the location of those points in the current frame.

```python
# calculate optical flow
p1, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame, p0, None, **lk_params)

# Select good points
good_new = p1[st==1]
good_old = p0[st==1]
```

## Calculating the Essential Matrix

After finding the locaions of corresponding points in two sequential frames, this information can be used to calculate the essential matrix. The essential matrix is based on the epipolar geometry describing the relative positions of the camera and contains the information needed to extract the rotation and translation from one camera frame to the next. For more information on epipolar geometry and the essential matrix, click [here](https://docs.opencv.org/3.1.0/da/de9/tutorial_py_epipolar_geometry.html).

```python
#Find the essential matrix
E, emask = cv2.findEssentialMat(oldPoints,newPoints,focal,pp,cv2.RANSAC,0.999,1.0)
```

## Calculating Pose/Updating Position

Now the essential matrix can be used to compute the rotation matrix and translation vector. We will use `cv2.recoverPose` to do this. There are four possible solutions when decomposing the essential matrix into R and t, and so `cv2.recoverPose` uses a Cheirality check to determine the solution that is geometrically feasible.

```python
#Find rotation and translation
points, R, t, pmask = cv2.recoverPose(E, oldPoints, newPoints)
t = np.asarray(t)
```

Now that we know the rotation and translation, we can use this to update our estimate of the robot's position. Before updating the position, we need to check to make sure the translation makes sense. Because we know that the robot only moves forward (not sideways or backwards), we can throw out any translations that do not have dominant motion in the forward direction.

```python
#Check if the translation makes sense
if t[2] > t[1] and t[2] > t[0] and t[2] > 0:
    #Concatenate R and t
    T = np.concatenate((R,t),axis=1)
    T = np.concatenate((T,lastRow),axis=0)

    #Update transformation matrix
    gk1 = np.dot(gk,T)

    position1[0][0] = gk1[0][3]
    position1[0][1] = gk1[1][3]

    #Increment
    gk = gk1
```

## Implementation

I implemented the visual odometry on the Turtlebot using ROS, with the help of [this](https://github.com/goromal/lab_turtlebot) code for controlling the Turtlebot and reading images from the camera (use the OpenCv branch). The only change needed is to call the `visualOdometry` function and the `pathControl` function from my code (available on Github [here](https://github.com/keenan573/visual-odometry)) in the `image_callback` function in the script `lab_turtlebot/turtlbot_cv/src/cv_command.py`.

## Results

One of the estimated paths found using this algorithm can be seen in the image below. The estimated path is certainly not perfect, but is able to show turns and the general direction taken by the robot. One challenge I had was a lack of features to track in the room where the testing was done, and the results would likely be better in a more feature-rich environment.

![alt text](https://github.com/keenan573/visual-odometry/images/pathcropped-01.png "One of the estimated paths")