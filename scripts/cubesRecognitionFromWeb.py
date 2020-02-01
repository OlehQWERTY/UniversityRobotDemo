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
	# _, ctrs, _ = cv2.findContours(img_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



	ctrs = cv2.findContours(img_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	ctrs = imutils.grab_contours(ctrs)
	# (ctrs, _) = contours.sort_contours(ctrs)  # ???



	# sort contours
	sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])

	for i, ctr in enumerate(sorted_ctrs):
		# Get bounding box
		x, y, w, h = cv2.boundingRect(ctr)

		# Getting ROI
		roi = image[y:y + h, x:x + w]

		# show ROI
		cv2.imshow('Region of Interest',roi)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.imshow('marked areas', image)


# cap = cv2.VideoCapture(0)

if __name__ == "__main__":
	while True:
		# _, frame = cap.read()

		frame = cv2.imread("../img_samples/1inBlack.png")
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
		lower_yellow = np.array([15, 100, 100]) #Lower boundary values for HSV
		upper_yellow = np.array([30, 255, 255]) #Upper boundary values for HSV

		yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow) # Threshold the HSV image to get only yellow colors
		opening_y = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel, iterations = 2) #Use morphology open to rid of false pos and false neg (noise)
		result_y = cv2.bitwise_and(frame, frame, mask = opening_y) #bitwise and the opening filter with the original frame 

		#Values for green during the day under a lamp 
		lower_green = np.array([45, 45, 10]) 
		upper_green = np.array([100, 255, 255])

		#Create filter for green
		green_mask = cv2.inRange(hsv, lower_green, upper_green)
		opening_g = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations = 1)
		result_g = cv2.bitwise_and(frame, frame, mask = opening_g)

		#Create filter for red
		lower_red1 = np.array([0, 70, 50])
		upper_red1 = np.array([10, 255, 255])
		lower_red2 = np.array([170, 70, 50])
		upper_red2 = np.array([180, 255, 255])
		mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
		mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
		red_mask = mask1 | mask2
		opening_r = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations = 2)
		result_r = cv2.bitwise_and(frame, frame, mask = opening_r)

		#Values for blue during the day under a lamp 
		lower_blue = np.array([110, 70, 30]) 
		upper_blue = np.array([130, 255, 255])

	# #Create filter for blue
		blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
		opening_b = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations = 1)
		result_b = cv2.bitwise_and(frame, frame, mask = opening_b)

		 #Tracking the color yellow
		# _, contours, _ = cv2.findContours(opening_y, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		contours, _ = cv2.findContours(opening_y.copy(), cv2.RETR_TREE,
		cv2.CHAIN_APPROX_SIMPLE)
		# contours = imutils.grab_contours(contours)
		# (contours, _) = contours.sort_contours(contours)

		print(contours[0])

		for y_pix, contour in enumerate (contours):
				area = cv2.contourArea (contour)
				if (area > 300):
						center(contours)
						x, y, w, h = cv2.boundingRect(contour)
						frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
						cv2.putText(frame, "Yellow", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 255))

		#Tracking the color green
		# _, contours, _ = cv2.findContours(opening_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours, _ = cv2.findContours(opening_y.copy(), cv2.RETR_TREE,
		cv2.CHAIN_APPROX_SIMPLE)

		for g_pix, contour in enumerate (contours):
				area = cv2.contourArea (contour)
				if (area > 300):
						center(contours)
						x, y, w, h = cv2.boundingRect(contour)
						frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (87, 139, 46), 2)
						cv2.putText(frame, "Green", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (87, 139, 46) )

		#Tracking the color blue
		# _, contours, _ = cv2.findContours(opening_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours, _ = cv2.findContours(opening_y.copy(), cv2.RETR_TREE,
		cv2.CHAIN_APPROX_SIMPLE)

		for b_pix, contour in enumerate (contours):
				area = cv2.contourArea (contour)
				if (area > 300):
						center(contours)
						x, y, w, h = cv2.boundingRect(contour)
						frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
						cv2.putText(frame, "Blue", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 0, 0) )

		#TODO Red 

		#gather selected colors to show only isolated colors 
		res = result_g + result_y + result_r + result_b

		#apply selected colors 
		#result = cv2.bitwise_and(frame, frame, mask = res)

		cv2.imshow("Detection and coordinates", frame)
		#cv2.imshow("Mask", green_mask)
		cv2.imshow("Colors Seen ", res)



		key = cv2.waitKey(1)
		if key == 27:
			break

# cap.release()
cv2.waitKey(0) # for single img test
cv2.destroyAllWindows()