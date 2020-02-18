import cv2
import numpy as np


from imutils import perspective
from imutils import contours
import imutils


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

	#Get binary image 
	_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

	#create structural element
	struc_ele = np.ones((5, 5), np.uint8)

	#Use Open Morphology
	img_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, struc_ele, iterations = 1)

	#find contours
	ctrs = cv2.findContours(img_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	ctrs = imutils.grab_contours(ctrs)

	# sort contours
	sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])

	for i, ctr in enumerate(sorted_ctrs):
		# Get bounding box
		x, y, w, h = cv2.boundingRect(ctr)

		# Getting ROI
		roi = image[y:y + h, x:x + w]

		# show ROI
		# cv2.imshow('Region of Interest',roi)  # white paper separated
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# cv2.imshow('marked areas', image)  # white paper maeked

if __name__ == "__main__":
	while True:

		frame = cv2.imread("../img_samples/1inBlack.png")

		find_ROI(frame)

		#Create kernel to use in filter
		kernel = np.ones((5, 5), np.uint8)

		bgr = [0, 247, 251]  # yellow
		# bgr = [206, 201,  99]  # blue

		thresh = 80
		 
		minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
		maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
		 
		maskBGR = cv2.inRange(frame,minBGR,maxBGR)

		# test?
		opening_BGR = cv2.morphologyEx(maskBGR, cv2.MORPH_OPEN, kernel, iterations = 2) #Use morphology open to rid of false pos and false neg (noise)
		# test?

		resultBGR_yellow = cv2.bitwise_and(frame, frame, mask = maskBGR)

		#Tracking the color yellow 
		contours, _ = cv2.findContours(opening_BGR.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


		for y_pix, contour in enumerate (contours):
				area = cv2.contourArea (contour)
				if (area > 300):
						center(contours)
						x, y, w, h = cv2.boundingRect(contour)
						frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
						cv2.putText(frame, "yellow", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255))

# ________________________________________________________________________

		# bgr = [0, 247, 251]  # yellow
		bgr = [206, 201,  99]  # blue

		thresh = 80
		 
		minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
		maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
		 
		maskBGR = cv2.inRange(frame,minBGR,maxBGR)

		# test?
		opening_BGR = cv2.morphologyEx(maskBGR, cv2.MORPH_OPEN, kernel, iterations = 2) #Use morphology open to rid of false pos and false neg (noise)
		# test?

		resultBGR_blue = cv2.bitwise_and(frame, frame, mask = maskBGR)

		#Tracking the color yellow 
		contours, _ = cv2.findContours(opening_BGR.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		for y_pix, contour in enumerate (contours):
				area = cv2.contourArea (contour)
				if (area > 300):
						center(contours)
						x, y, w, h = cv2.boundingRect(contour)
						frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
						cv2.putText(frame, "blue", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 0, 0))


		cv2.imshow("Detection and coordinates", frame)
		
		cv2.imshow("resultBGR_yellow ", resultBGR_yellow)
		cv2.imshow("resultBGR_blue ", resultBGR_blue)

		cv2.imshow("resultBGR_blue+yellow ", resultBGR_blue + resultBGR_yellow)


		key = cv2.waitKey(1)
		if key == 27:
			break

cv2.waitKey(0) # for single img test
cv2.destroyAllWindows()