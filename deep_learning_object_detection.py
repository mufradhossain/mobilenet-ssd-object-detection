# import the necessary packages
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)

def after(our_image):

	image = np.array(our_image)
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	result=[]
	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > 0.2:
			# extract the index of the class label from the `detections`,
			# then compute the (x, y)-coordinates of the bounding box for
			# the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# display the prediction
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			result.append(str(label))
			cv2.rectangle(image, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(image, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	#return the output image
	return image , result



def main():

	hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
	st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
	st.title("mobilenet-ssd")
	html_temp = """
	<body style="background-color:red;">
	<div style="background-color:#FE615A ;padding:10px">
	<h2 style="color:white;text-align:center;">Object Detection Based on the Mobilenet-ssd Model</h2>
	</div>
	</body>
	"""
	st.markdown(html_temp, unsafe_allow_html=True)

	image_file = st.file_uploader("Upload Image to Detect Objects", type=['jpg', 'png', 'jpeg'])
	if image_file is not None:

		imag = Image.open(image_file)
		st.image(image_file)

	if st.button("Detect Objects"):
		res , cap=after(imag)
		st.image(res)
		st.text("Objects: " + str(cap))

if __name__ == '__main__':
    main()