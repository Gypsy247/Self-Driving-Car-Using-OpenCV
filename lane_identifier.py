"""

The 'test2.mp4' file used in this code is available in the 'The Complete Self-Driving Car Course - Applied Deep Learning' course.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    # average coordinates of the lines on the left
    left_fit = []
    # average coordinates of the lines on the right
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # parameters: x coordinates, y coordinates, degree of the polynomial
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


'''
canny(image):

We begin by converting the image to grayscale, which makes the process less computationally intensive.
This is due to the fact that each pixel in a coloured image is a combination of three channels (the colours red, green and blue),
whereas Grayscale is just a single channel with values ranging in intensity from 0 t 255.

Gaussian Blur is used to take weighted average of neighbouring pixels which in turn helps reduce noise.
This is done using a kernel of normally distributed numbers

Edge detection is done using Canny, which takes the derivative of the pixel in all directions which gives the gradient and
then maps out a line of the the strongest gradient which forms our separation line.

'''


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


'''
Drawing the detected lines over a base image to display the superimposed image

'''


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        # We then unpack the array into four coordinates
        for x1, y1, x2, y2 in lines:
            # parameters: image, pt1, pt2, BGR colour scheme, line thickness
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


'''
region_of_interest(image):

This function helps define the area of the image that we are concerned with. In our case its a polygonal area
that is roughly shaped like a triangle and contains the lanes of the road. So we isolate that area and create
a mask over it with a contrasting colour to the background. So a white mask on black background.

'''


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    # parameters: image, polar resolution(row, theta), threshold, place holder array, min. line length, max gap for segmented lines
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,
                            np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    # parameters: base image, intensity, upper image, intensity, scalar/gamma value
    merged_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    # displaying the image
    cv2.imshow("result", merged_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
