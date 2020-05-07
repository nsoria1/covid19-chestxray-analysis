import os
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import cv2
import keras
import numpy as np
import h5py
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from keras import backend as K
UPLOAD_FOLDER = './Uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
#app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			image = cv2.imread(os.path.dirname(os.path.realpath(__file__))+"/uploads/"+filename)
			result = covidOrNot(image)
			redirect(url_for('upload_file',filename=filename))
			return '''
			<!doctype html>
			<title>Results</title>
			<h1>Image contains a '''+result+'''</h1>
			<form method=post enctype=multipart/form-data>
			  <input type=file name=file>
			  <input type=submit value=Upload>
			</form>
			'''
	return '''
	<!doctype html>
	<title>Upload an image</title>
	<h1>Upload an image</h1>
	<form method=post enctype=multipart/form-data>
	  <input type=file name=file>
	  <input type=submit value=Upload>
	</form>
	'''
def covidOrNot(image):
	'''Determines if the chest x ray has covid-19 or not'''
	classifier = load_model('./Models/covid_model.h5')
	image = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
	image = image.reshape(1,224,224,3) 
	res = str(classifier.predict(image, 1, verbose = 0)[0][0])
	print(res)
	print(type(res))
	if res == "0":
		res = "Patient who does not have COVID-19"
	else:
		res = "Patient who has COVID-19"
	K.clear_session()
	return res
	
if __name__ == "__main__":
	app.run()