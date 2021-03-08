# import the necessary packages
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

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
	perres=[]
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
			label = "{}".format(CLASSES[idx])
			per = confidence*100
			perres.append(round(per,2))
			
			result.append(str(label))
			cv2.rectangle(image, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(image, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	#return the output image
	return image , result , perres

def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

def main():

	_max_width_()

	hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
	st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

	#st.title("mobilenet-ssd")
	html_temp = """
	<body style="background-color:red;">
	<div style="background-color:#FACA2B ;padding:10px">
	<h2 style="color:white;text-align:center;">Object Detection Based on the Mobilenet-ssd Model</h2>
	</div>
	</body>
	"""
	st.markdown(html_temp, unsafe_allow_html=True)
	st.set_option('deprecation.showPyplotGlobalUse', False)

	image_file = st.file_uploader("Upload Image to Detect Objects", type=['jpg', 'png', 'jpeg'])
	if image_file is not None:

		col1, col2, col3 = st.beta_columns(3)
		col1.header("Image")
		col1.image(image_file)
		imag = Image.open(image_file)
		res , cap, per=after(imag)
		col2.header("Objects")
		col2.image(res)

		newcap = [] 
		newper=[]
		bcolor=[]
		j=0
		for item in cap:
			if item not in newcap: 
				newcap.append(item) 
				newper.append(per[j])
				if (per[j]<40):
					bcolor.append("red")
				elif (per[j] >40 and per[j] < 60):
					bcolor.append("orange")
				elif (per[j] > 60 and per[j]<80):
					bcolor.append("yellow")
				else:
					bcolor.append("green")

			j+=1

		objclasses = newcap		
		outofhund = []

		plt.figure(figsize=(10,8))

		for x in newper:
			outofhund.append(100-x)
		#plt.barh(newcap, newper, color="#238823")  
		plt.barh(newcap, newper, color=bcolor)  
		plt.barh(newcap,outofhund, left=newper , color="grey")
		for i, v in enumerate(newper):
			plt.text(v -15, i-0.05, str(v)+"%", color='black', fontweight='bold')
		plt.xlabel('Confidence Level')  
		plt.rcParams['ytick.labelsize']=16
		plt.yticks(rotation=90)
		plt.ylabel('Objects')
		col3.header("Confidence")
		col3.pyplot(use_column_width=True)

if __name__ == '__main__':
    main()