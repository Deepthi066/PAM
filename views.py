from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import HttpResponse,JsonResponse,request
from django.template import loader
import os
import json
import numpy as np
from keras.models import load_model
from twilio.rest import Client

account_sid="ACe03a3f8d9483cad457f29f83425501f7"
auth_token="e173bf9393ec8d99cfd4fc8c74e5cb60"
client = Client(account_sid, auth_token)


def send_sms_notification(body):
  message = client.messages.create(
      body=body,
    from_="+12513123406",
    to="+917019088298",
  )
  print(message.sid)


model_failure = load_model('../../../models/deeplearning-neuralnetworks/machine_failure_prediction_model.h5')

# Load the trained failure type prediction model
model_type = load_model('../../../models/deeplearning-neuralnetworks/machine_failure_type_prediction_model.h5')


# Define a function to make predictions
def predict_failure(input_data):
    # Use the machine failure prediction model to make predictions
    prediction = model_failure.predict(input_data)

    predicted_class = np.argmax(prediction)
    # Return the prediction
    return predicted_class

def predict_type(input_data):
    # Use the failure type prediction model to make predictions
    prediction = model_type.predict(input_data)
    predicted_class = np.argmax(prediction)
    # Return the prediction
    return predicted_class

def preprocess_input(temperature, process_temperature, torque, tool_wear):

    temp_diff = temperature - process_temperature
    
    # Return the preprocessed input as a NumPy array
    return np.array([[temperature, process_temperature, torque, tool_wear,temp_diff ]])

# Create your views here.

def index(request):
    template=loader.get_template('index.html')
    return HttpResponse(template.render({}, request))

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            # Access JSON data from the request body
            data = json.loads(request.body)

            # Access individual JSON fields
            air_temp = data.get('airTemp')
            process_temp = data.get('processTemp')
            rotational_speed = data.get('rotationalSpeed')
            torque = data.get('torque')
            tool_wear = data.get('toolWear')

            air_temp = int(air_temp)
            process_temp = int(process_temp)
            rotational_speed = int(rotational_speed)
            torque = int(torque)
            tool_wear = int(tool_wear)
            #print(data)

            # Perform prediction or other logic using the data

            input_data = preprocess_input(air_temp,process_temp, torque, tool_wear)
    
            # Make predictions
            fail_prediction = predict_failure(input_data)
            if fail_prediction == 0:
                failure_prediction="No Failure"
            else:
                failure_prediction="Failure"

            

            failure_types=["Tool Wear Failure (TWF)","Heat Dissipation Failure (HDF)","Power Failure (PWF)","Overstrain Failure (OSF)","Random Failure (RNF)"]
            type_prediction = predict_type(input_data)
            type_prediction=type_prediction-1
            prediction_class=failure_types[type_prediction]

            body=f"Prediction : {failure_prediction} \nFailure Type : {prediction_class}"

            send_sms_notification(body)


            # Return a JSON response
            return JsonResponse({'prediction': failure_prediction,'type': prediction_class})
        except json.JSONDecodeError as e:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
        