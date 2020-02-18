#python

# link: https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/

import numpy as np
import cv2
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
 
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
		# print(x,y)
		# print(refPt[0][0])

		image = clone.copy()  # automatic clearing from previous moussing

		print('x: ' + str(refPt[0][0]) + ' y: ' + str(refPt[0][1]))
				
		brg = []
		brg = image[refPt[0][1], refPt[0][0]]
		# brg = [40, 158, 16]
		thresh = 80

		print('brg: ' + str(brg))
		print("thresh: " + str(thresh))
		print("______________________")

		imgFiltered(brg, thresh)
		cv2.rectangle(image, (refPt[0][0] - 1, refPt[0][1] - 1), (refPt[0][0] + 1, refPt[0][1] + 1), (0, 255, 0), 2)
		cv2.imshow("image", image)



def imgFiltered(bgr, thresh):
	bright = image
	brightHSV = image
	brightYCB = image
	brightLAB = image

	# bgr = [40, 158, 16]
	# thresh = 80
	 
	minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
	maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
	 
	maskBGR = cv2.inRange(bright,minBGR,maxBGR)
	resultBGR = cv2.bitwise_and(bright, bright, mask = maskBGR)
	 
	#convert 1D array to 3D, then convert it to HSV and take the first element 
	# this will be same as shown in the above figure [65, 229, 158]
	hsv = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2HSV)[0][0]
	 
	minHSV = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
	maxHSV = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])
	 
	maskHSV = cv2.inRange(brightHSV, minHSV, maxHSV)
	resultHSV = cv2.bitwise_and(brightHSV, brightHSV, mask = maskHSV)
	 
	#convert 1D array to 3D, then convert it to YCrCb and take the first element 
	ycb = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2YCrCb)[0][0]
	 
	minYCB = np.array([ycb[0] - thresh, ycb[1] - thresh, ycb[2] - thresh])
	maxYCB = np.array([ycb[0] + thresh, ycb[1] + thresh, ycb[2] + thresh])
	 
	maskYCB = cv2.inRange(brightYCB, minYCB, maxYCB)
	resultYCB = cv2.bitwise_and(brightYCB, brightYCB, mask = maskYCB)
	 
	#convert 1D array to 3D, then convert it to LAB and take the first element 
	lab = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2LAB)[0][0]
	 
	minLAB = np.array([lab[0] - thresh, lab[1] - thresh, lab[2] - thresh])
	maxLAB = np.array([lab[0] + thresh, lab[1] + thresh, lab[2] + thresh])
	 
	maskLAB = cv2.inRange(brightLAB, minLAB, maxLAB)
	resultLAB = cv2.bitwise_and(brightLAB, brightLAB, mask = maskLAB)
	 
	cv2.imshow("Result BGR", resultBGR)
	# cv2.imshow("Result HSV", resultHSV)
	# cv2.imshow("Result YCB", resultYCB)
	# cv2.imshow("Output LAB", resultLAB)
	# cv2.waitKey(0) # for single img test


image = cv2.imread("../img_samples/1inBlack.png")
# image = cv2.imread("../img_samples/maxresdefault.jpg")
clone = image.copy()
# bright = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)



cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)  # mouse press processing

cv2.imshow("image", image)

while True:
	key = cv2.waitKey(1) & 0xFF
	# if the 'c' or 'ESC' key is pressed, break from the loop
	if key == ord("c") or key == 27:
		break

	# close window with win "X" button
	xPos, yPos, width, height = cv2.getWindowImageRect("image")
	if xPos == -1: # if user closed window
		break # do whatever you want here if the user clicked CLOSE

# close all open windows
cv2.destroyAllWindows()