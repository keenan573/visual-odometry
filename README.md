# Visual Odometry

This document will explain the theory behind visual odometry and show an implementation that can be used on a Turtlebot using Python and OpenCV.

## Visual Odometry Overview

Visual odometry allows you to determine a path traveled using only image data from a camera. In this example we will use a single calibrated camera.

The following steps will be described:

-  Feaure detection
-  Optical flow
-  Calculating the essential matrix
-  Calculating pose/updating position

# Feature Detection

The first step is to detect features in the image that can be tracked. OpenCV has several ways of doing this, but here we will use the FAST feature detector.

```python
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find the keypoints
kp = fast.detect(frame1,None)
# Extract the x and y coordinates from the keypoints
p0 = np.float32([p.pt for p in kp])
p0 = p0.reshape(-1,1,2)
```