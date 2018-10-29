from flask import Flask, jsonify
from flask import request, redirect
import numpy as np
import pandas as pd 
import json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib


model = joblib.load('prediction.pkl')

var_list = ('a1', 'a2', 'a3', 'a4', 'a5', 'a6', 
'l1', 'l2', 'l3', 'l4', 'au1', 'au2', 'au3', 'au4', 'm1', 'm2', 'm3', 
'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'angry', 'scared', 
'frust', 'disill', 'hope', 'excited', 'conf','s_1', 
's_2', 's_3', 'race', 'educ', 'political','V1', 'V4', 'V5','trust', 'self_effic', 'rigged', 
'stranger', 'direction', 'pewb','leader', 'threat', 'news', 'pro_soc')


app = Flask(__name__)

@app.route('/predict_segment')

def long_prediction():
	# read the inputs from query string
	query_string = request.args
	d = query_string.to_dict()

	# check inputs 
	for key in d:
		try:
			d[key] = int(d[key])
		except:
			raise InvalidUsage("inputs have to be integer", status_code = 410)

	# check if inputs are in query string

	for key in inputs_long:
		if key not in d: raise InvalidUsage("Input missing", status_code = 410)

	# check if inputs are in proper range:
	for key in inputs_long:
		if d[key] < 0 or d[key] > 5:
			raise InvalidUsage("variables should be in range[0,5]", status_code = 410)


	# perform the prediction model
	inputs = pd.DataFrame([d])
	inputs = inputs[var_list]
	prediction = model.predict(inputs).tolist()

	# return prediction result
	return redirect(get_url(prediction))


# error handler
class InvalidUsage(Exception):
	status_code = 400


	def __init__(self, message, status_code=None, payload=None):
		Exception.__init__(self)
		self.message = message
		if status_code is not None:
			self.status_code = status_code
			self.payload = payload


	def to_dict(self):
		rv = dict(self.payload or ())
		rv['message'] = self.message
		return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
	response = jsonify(error.to_dict())
	response.status_code = error.status_code
	return response



def get_url(prediction):
	if prediction == 1:
		w = 'http://www.google.com'
	elif prediction == 2:
		w = 'http://www.cnn.com'
	elif prediction == 3:
		w = 'http://www.espn.com'
	return w


