import cv2
import numpy as np

count = 1
frame1 = np.zeros((480,640,3))
p0 = []
#Starting position
position = np.array([[0.0,0.0,0.0]])
position1 = np.array([[0.0,0.0,0.0]])

#Last row for transformation matrix
lastRow = np.array([[0.0, 0.0, 0.0, 1.0]])

#Gk starts out as identity, position, lastrow
gk = np.concatenate((np.identity(3,dtype=float),position.T),axis=1)
gk = np.concatenate((gk,lastRow),axis=0)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#Make a blank image to draw path on
image = np.zeros((1000, 1000, 3), np.uint8)
pathmask = np.zeros_like(image)

def visualOdometry(frame):
    global count
    global frame1
    global p0

    if count == 1:
        #Save the first frame
        frame1 = frame
        #Convert to grayscale
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        #Crop image to top half of frame
        frame1 = frame1[120:540,:]

        # Initiate FAST object with default values
        fast = cv2.FastFeatureDetector_create()

        # find the keypoints
        kp = fast.detect(frame1,None)
        p0 = np.float32([p.pt for p in kp])
        p0 = p0.reshape(-1,1,2)

        count = 2
        return
    else:
        #Convert the frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Crop image to top half of frame
        frame = frame[120:540,:]
        #Calculate the optical flow
        good_new,good_old = opticalFlow(frame)
        
        #If feature count is too small, find new features
        if len(good_new)<100:
            # Initiate FAST object with default values
            fast = cv2.FastFeatureDetector_create()
            # find the keypoints
            kp = fast.detect(frame,None)

            p0 = np.float32([p.pt for p in kp])
            p0 = p0.reshape(-1,1,2)

        #Update the current position
        calcPosition(good_old,good_new)
        #Draw the path
        drawPath()

def opticalFlow(frame):
    global frame1
    global p0

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the feature points
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        frame = cv2.circle(frame,(a,b),5,(0,0,0),1)

    #Update frames
    frame1 = frame
    #update points
    p0 = good_new.reshape(-1,1,2)

    return (good_new,good_old)


def calcPosition(oldPoints,newPoints):
    global gk

    #Focal length and principle point
    focal = 527
    pp = (327.72,247.72)

    #Find the essential matrix
    E, emask = cv2.findEssentialMat(oldPoints,newPoints,focal,pp,cv2.RANSAC,0.999,1.0)

    #Find rotation and translation
    points, R, t, pmask = cv2.recoverPose(E, oldPoints, newPoints)
    t = np.asarray(t)

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

def drawPath():
    global position
    global position1
    global image
    global pathmask

    #Add 500 so that the path starts in the middle of the image
    a = int(position[0][0]*2) + 500
    b = int(position[0][1]*2) + 500

    c = int(position1[0][0]*2) + 500
    d = int(position1[0][1]*2) + 500

    #draw the path
    pathmask = cv2.line(pathmask, (a,b),(c,d), [0,0,255], 2)
    pathimage = cv2.add(image,pathmask)

    cv2.imshow('path',pathimage)
    cv2.imshow('frame',frame1)

    #Increment postion
    position = position1

def pathControl():
    global count

    v = 0.0
    w = 0.0

    if count < 250:
        v = 0.1
        w = 0.0
    elif count < 298:
        v = 0.1
        w = 0.5
    elif count < 550:
        v = 0.1
        w = 0.0
    elif count < 598:
        v = 0.1
        w = 0.5
    elif count < 750:
        v = 0.1
        w = 0.0
    elif count < 798:
        v = 0.1
        w = 0.5
    else:
        v = 0.1
        w = 0.0

    count = count + 1

    return (v,w)
