from __future__ import print_function, unicode_literals
import warnings
from flask import Flask, request
import time, os, requests, json
import tensorflow as tf
import cv2
import numpy as np

# interpreter = tf.lite.Interpreter(model_path="model.tflite")

# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# interpreter.allocate_tensors()




warnings.filterwarnings("ignore")
app = Flask(__name__)



@app.route('/', methods=['GET'])
def success():
    return "success"
    
@app.route('/ping', methods=['GET'])
def ping():
    return "true"

@app.route('/inference', methods=['POST', 'GET'])
def main():
    # st_tim = time.time()
    # Fetching the variables from the post request
    print(request.data)
    param = request.data 		#.args['param']
    param = json.loads(param.decode('utf-8'))
    image_file = param['image_path']
    interpreter = tf.lite.Interpreter(model_path="model_convert_test_tf114.tflite")

	# Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()

    img = cv2.imread(r"{}".format(image_file))
    new_img = cv2.resize(img, (224, 224))

    new_img = np.array(new_img, dtype=np.float32)
	# input_details[0]['index'] = the index which accepts the input
    interpreter.set_tensor(input_details[0]['index'], [new_img])


	# run the inference
    interpreter.invoke()

	# output_details[0]['index'] = the index which provides the input
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print("For file {}, the output is {}".format(image_file, output_data))
    a={'Crack_and_Wrinkle_defect':float(output_data[0][0])*255,
    'Healthy':float(output_data[0][1])*255}
    return json.dumps(a)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000,debug=True)
