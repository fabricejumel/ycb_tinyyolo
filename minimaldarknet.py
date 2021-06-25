import numpy as np
import time
import cv2




class ObjectDetection(object):

	def __init__(self):
		# Get real-time video stream through opencv

		LABELS_FILE='ycb_simu.names'
		CONFIG_FILE='yolov3-tiny-ycb_simu_test.cfg'
		WEIGHTS_FILE='yolov3-tiny-ycb_simu_best_004.weights'
		self.CONFIDENCE_THRESHOLD=0.3

		self.H=None
		self.W=None

		self.LABELS = open(LABELS_FILE).read().strip().split("\n")
		np.random.seed(4)
		self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),	dtype="uint8")

		self.net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
		
		self.ln = self.net.getLayerNames()
		self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]


	def detect(self,image):
			#image = self.video.read()
			# Treatment
			data=[]
			blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
			self.net.setInput(blob)
			if self.W is None or self.H is None:
				(self.H, self.W) = image.shape[:2]

			layerOutputs = self.net.forward(self.ln)






			# initialize our lists of detected bounding boxes, confidences, and
			# class IDs, respectively
			boxes = []
			confidences = []
			classIDs = []

			# loop over each of the layer outputs
			for output in layerOutputs:
				# loop over each of the detections
				for detection in output:
					# extract the class ID and confidence (i.e., probability) of
					# the current object detection
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					# filter out weak predictions by ensuring the detected
					# probability is greater than the minimum probability
					if confidence > self.CONFIDENCE_THRESHOLD:
						# scale the bounding box coordinates back relative to the
						# size of the image, keeping in mind that YOLO actually
						# returns the center (x, y)-coordinates of the bounding
						# box followed by the boxes' width and height
						box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
						(centerX, centerY, width, height) = box.astype("int")

						# use the center (x, y)-coordinates to derive the top and
						# and left corner of the bounding box
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						# update our list of bounding box coordinates, confidences,
						# and class IDs
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)

			# apply non-maxima suppression to suppress weak, overlapping bounding
			# boxes
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD,
				self.CONFIDENCE_THRESHOLD)

			# ensure at least one detection exists
			if len(idxs) > 0:
				# loop over the indexes we are keeping
				for i in idxs.flatten():
					# extract the bounding box coordinates
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])


					tuple_i=tuple([x,y,x+w,y+h,self.LABELS[classIDs[i]],confidences[i]])
					data.append(tuple_i)

			# print(data)
			return  data 

if __name__ == '__main__' :
	print("direct test of ycb tiny yolo v0.0.4")
	ycb=ObjectDetection()
	img=cv2.imread("test_ycb.jpg")
	output=ycb.detect(img)
	print(output)
