import cv2
import numpy as np



from imutils import perspective
from imutils import contours
import imutils



# frame = cv2.imread("../img_samples/1.png")
# cv2.imshow("Over the Clouds - gray", frame)
# cv2.waitKey(0)


#find the center of an object
def center(contours):
	# calculate moments for each contour
	for c in contours:
		M = cv2.moments(c)

		if M["m00"] != 0:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
		else:
			cX, cY = 0, 0
		tempStr = str(cX) + ", " + str(cY)
		cv2.circle(frame, (cX, cY), 1, (0, 0, 0), -1) #make a dot at the center of the object 
		cv2.putText(frame, tempStr, (cX - 25, cY - 25),cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 0, 0), 1) #print the coordinates on the image

#get the region of interest
def find_ROI(frame):
	image = frame.copy()

	#change to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# cv2.imshow("gray", gray)
	# cv2.waitKey(0)


	#Get binary image 
	# _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
	_, thresh = cv2.threshold(gray, 195, 255, cv2.THRESH_BINARY)

	# cv2.imshow("threshold", thresh)
	# cv2.waitKey(0)

	#create structural element
	struc_ele = np.ones((5, 5), np.uint8)

	#Use Open Morphology
	img_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, struc_ele, iterations = 1)

	# cv2.imshow("morphologyEx", img_open)
	# cv2.waitKey(0)

	#find contours
	# _, ctrs, _ = cv2.findContours(img_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	ctrs = cv2.findContours(img_open.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
	ctrs = imutils.grab_contours(ctrs)
	(ctrs, _) = contours.sort_contours(ctrs)
	# print(ctrs)


	# sort contours
	sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])

	for i, ctr in enumerate(sorted_ctrs):
		# Get bounding box
		x, y, w, h = cv2.boundingRect(ctr)

		# Getting ROI
		roi = image[y:y + h, x:x + w]

		# show ROI
		# cv2.imshow('Region of Interest',roi)

		# cv2.waitKey(0) # for single img test

		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# print(i)
		# print(ctr)
	# cv2.imshow('marked areas', image)

	# cv2.waitKey(0) # for single img test



# cap = cv2.VideoCapture(0)

if __name__ == "__main__":
	# while True:
	# _, frame = cap.read()
	# frame = cv2.imread("../img_samples/1.png")
	frame = cv2.imread("../img_samples/1inBlack.png")
	# cv2.imshow("Show input img", frame)
	# cv2.waitKey(0)
	# cap = cv2.VideoCapture(1) 
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	find_ROI(frame)
	# l_h = cv2.getTrackbarPos("L - H", "Trackbars")
	# l_s = cv2.getTrackbarPos("L - S", "Trackbars")
	# l_v = cv2.getTrackbarPos("L - V", "Trackbars")
	# u_h = cv2.getTrackbarPos("U - H", "Trackbars")
	# u_s = cv2.getTrackbarPos("U - S", "Trackbars")
	# u_v = cv2.getTrackbarPos("U - V", "Trackbars")

	#Create kernel to use in filter
	kernel = np.ones((5, 5), np.uint8)

	#Create filter for yellow
	# lower_yellow = np.array([15, 100, 100]) #Lower boundary values for HSV
	# upper_yellow = np.array([30, 255, 255]) #Upper boundary values for HSV

	lower_yellow = np.array([0, 150, 100]) #Lower boundary values for HSV
	upper_yellow = np.array([255, 255, 255]) #Upper boundary values for HSV

	yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow) # Threshold the HSV image to get only yellow colors
	# opening_y = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel, iterations = 2) #Use morphology open to rid ...
	opening_y = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel, iterations = 2) #Use morphology open to rid ...
	# print(len(yellow_mask))  # All px belong to cubes

	# cv2.imshow('opening_y', opening_y)

# copy from ROI
	ctrs = cv2.findContours(opening_y.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
	ctrs = imutils.grab_contours(ctrs)
	(ctrs, _) = contours.sort_contours(ctrs)
	# print(ctrs)

	sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])
	print(sorted_ctrs)

	numb_of_window = 0
	for i, ctr in enumerate(sorted_ctrs):
		# Get bounding box
		x, y, w, h = cv2.boundingRect(ctr)

		# Getting ROI
		roi = frame[y:y + h, x:x + w]

		# show ROI
		numb_of_window = numb_of_window + 1
		cv2.imshow('Region of Interest' + str(numb_of_window),roi)

		# cv2.waitKey(0) # for single img test

		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# print(i)
		# print(ctr)
	cv2.imshow('marked areas', frame)

# copy from ROI

	cv2.waitKey(0) # for single img test