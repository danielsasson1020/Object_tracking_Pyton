import numpy as np
import cv2
import time


# roi = region of interest
def select_roi(event, x, y, flags, param):
    global frame, roiPts, inputMode

    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 2:
        roiPts.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow(window_name, frame)


# for the track bar callback
def nothing(x):
    pass


# initializing variables
frame = None
avg = None
roiPts = []
inputMode = False
x0, y0, w0, h0 = 0, 0, 0, 0
window_name = "Object Tracking"

video_name = 'video2.avi'
# example videos with recommended values for optimal results
# video1.avi delta_threshold = 8 avg_weight = 0.08 search range = 100 (CT)
# video2.avi delta_threshold = 25 avg_weight = 0.10 search range = 110  (helicopter)
# video3.avi delta_threshold = 20 avg_weight = 0.10 search range = 100 (cars)
# video4.avi delta_threshold = 15 avg_weight = 0.10 search range = 100 (tennis)

cap = cv2.VideoCapture(video_name)
cv2.namedWindow(window_name, flags=cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(window_name, select_roi)

# delta_threshold = Smaller values will lead to more motion being detected tho can lead to noisy tracking,
# larger values to less motion detected and less noise.
cv2.createTrackbar("Threshold", window_name, 18, 50, nothing)

# avg_weight = Smaller values will make the avg background update slower, making it easier to follow slow moving objects
# but will lead to more "noisy" tracking
# higher values will make the background update faster which helps tracking fast moving objects
cv2.createTrackbar("Avg Weight", window_name, 10, 20, nothing)

# determines the range in which looking to update the single object tracking rectangle
cv2.createTrackbar("Search Range", window_name, 100, 200, nothing)

# determines how fast the video plays
cv2.createTrackbar("Delay", window_name, 5, 50, nothing)

# change these with caution
min_area = 5
erode_strength = 2
dilate_strength = 15

while cap.isOpened():
    ret, frame = cap.read()

    # if no frame then we reached end of the video, so can replay it
    if frame is None:
        cap = cv2.VideoCapture(video_name)
        avg = None
        continue

    # resizing for convenience
    frame = cv2.resize(frame, (640, 480))

    # if user pressed "t" then switch to selection mode
    # in which user needs to select the top-left and bottom-right corners of an object to track
    key = cv2.waitKey(1) & 0xFF
    if key == ord("t") and inputMode is False:
        inputMode = True
        while len(roiPts) < 2:
            cv2.imshow(window_name, frame)
            cv2.waitKey(0)
        x0, y0, w0, h0 = roiPts[0][0], roiPts[0][1], roiPts[1][0] - roiPts[0][0], roiPts[1][1] - roiPts[0][1]

    # press "c" to clear the object to track
    if key == ord("c") and inputMode is True:
        inputMode = False
        roiPts = []
        x0, y0, w0, h0 = 0, 0, 0, 0

    delta_threshold = cv2.getTrackbarPos("Threshold", window_name)
    avg_weight = cv2.getTrackbarPos("Avg Weight", window_name) / 100.0

    # operations on the frame,
    # resizing it and making a new frame to work with
    # which is in gray scale and blurred to remove noise
    # blur on 21x21 region
    nextFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nextFrame = cv2.GaussianBlur(nextFrame, (21, 21), 0)

    # in the first run, need to initialize the avg frame background
    if avg is None:
        avg = nextFrame.copy().astype("float")
        continue

    # updates the background by updating the average frame
    cv2.accumulateWeighted(nextFrame, avg, avg_weight)

    # calculates the difference between the new frame and the average background
    frameDelta = cv2.absdiff(nextFrame, cv2.convertScaleAbs(avg))

    # making a new frame from the difference frame in which all the pixels above
    # a certain value will be increased to 255
    # binary = either 255 or 0
    threshold = cv2.threshold(frameDelta, delta_threshold, 255, cv2.THRESH_BINARY)[1]

    # erode the threshold frame, making objects thinner to remove noise
    # and after that dilate the objects with a number of iterations to make them bigger and easier to track
    # closing small holes inside the object
    kernel = np.ones((5, 5), np.uint8)
    threshold = cv2.dilate(threshold, None, iterations=dilate_strength)
    threshold = cv2.erode(threshold, kernel, iterations=erode_strength)

    # finding the contours of all the objects in the threshold frame
    # and returning them as a list, each contour in the list is numpy array of coordinates
    # retrieval mode = RETR_EXTERNAL
    # approximation method = CHAIN_APPROX_SIMPLE
    _, contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for each contour, if its larger than a set min_area value
    # then draw a green rectangle around it
    draw_static = True
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue

        (x, y, w, h) = cv2.boundingRect(c)

        # if in single object tracking mode then need to update the tracking rectangle
        # according to the range of all the contours, for which the contour inside the range will be chosen
        # if didn't find any then draw the last rectangle place
        if inputMode is True:
            rangeX = abs(x-x0)
            rangeY = abs(y-y0)
            rangeW = abs(w-w0)
            rangeH = abs(h-h0)

            search_range = cv2.getTrackbarPos("Search Range", window_name)
            if rangeX < search_range and rangeY < search_range and rangeW < search_range and rangeH < search_range:
                x0 = x
                y0 = y
                w0 = w
                h0 = h
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                draw_static = False
                break

        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            draw_static = False

    if draw_static:
        cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)

    # show the result to the user
    time.sleep(cv2.getTrackbarPos("Delay", window_name) / 100.0)
    cv2.putText(frame, "Press ESC To Exit.", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.imshow(window_name, frame)
    cv2.imshow('Threshold', threshold)
    # cv2.imshow('Frame Delta', frameDelta)

    # 27 is ESC ascii value
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
