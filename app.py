import pandas as pd
import numpy as np

#########
# from OpenSSL import SSL

# context = SSL.Context(SSL.TLSv1_2_METHOD)
# context.use_privatekey_file('/etc/letsencrypt/live/messyplayground.com/privkey.pem')
# context.use_certificate_chain_file('/etc/letsencrypt/live/messyplayground.com/fullchain.pem')
# context.use_certificate_file('/etc/letsencrypt/live/messyplayground.com/cert.pem')
#########

# Assuming the CSV file is in the same directory as your server script
# data = pd.read_csv('./X_train.csv')
# data = pd.read_csv('./X_t.csv')

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model('./project_bestmodel.h5')
column_name = ['First Term Gpa',
    'Second Term Gpa',
    'Age Group',
    'High School Average Mark',
    'Math Score',
    'English Grade',
    'First Language_1',
    'First Language_2',
    'First Language_3',
    'Funding_1',
    'Funding_2',
    'Funding_4',
    'Funding_5',
    'Funding_8',
    'Funding_9',
    'School_6',
    'FastTrack_1',
    'FastTrack_2',
    'Coop_1',
    'Coop_2',
    'Residency_1',
    'Residency_2',
    'Gender_1',
    'Gender_2',
    'Gender_3',
    'Previous Education_0',
    'Previous Education_1',
    'Previous Education_2']

cat_map = {0: "fatal", 1: "non-fatal"}

@app.route('/predict', methods=['POST'])
def predict():
    data = pd.DataFrame(0, index=np.arange(1), columns=column_name) 
    # print(data)
    # print(type(data))
    # print("\n\n\n\n\n")

    request_data = request.json
    # print(request_data)
    # print(request_data.get('second_gpa', None))
    # print("\n\n\n\n\n")

    data['First Term Gpa'] = request_data.get('first_gpa', None)
    data['Second Term Gpa'] = request_data.get('second_gpa', None)
    data['First Language_' + request_data.get('first_language', None)] = 1
    if request_data.get('funding', None) == 1 or \
        request_data.get('funding', None) == 2 or \
        request_data.get('funding', None) == 4 or \
        request_data.get('funding', None) == 5 or \
        request_data.get('funding', None) == 8 or \
        request_data.get('funding', None) == 9:
        data['Funding_' + request_data.get('funding', None)] = 1
    if request_data.get('school', None) == 6:
        data['School_6'] = 1
    data['FastTrack_' + request_data.get('fast_track', None)] = 1
    data['Coop_' + request_data.get('coop', None)] = 1
    data['Residency_' + request_data.get('residency', None)] = 1
    data['Gender_' + request_data.get('gender', None)] = 1
    data['Previous Education_' + request_data.get('prev_edu', None)] = 1
    data['Age Group'] = request_data.get('age_gp', None)
    data['High School Average Mark'] = request_data.get('highschool_avg', None)
    data['Math Score'] = request_data.get('math_score', None)
    data['English Grade'] = request_data.get('eng_grade', None)


    # return jsonify("123")
    # return jsonify(data.shape)

    # Preprocess the data as needed
    # processed_data = preprocess(data)  # Implement this function based on your model's needs
    # processed_data = data  # Implement this function based on your model's needs
    # return jsonify(processed_data)

    # Make predictions
    predictions = model.predict(data.astype('float32'))

    # Convert predictions to a suitable format for JSON response
    # response = format_predictions(predictions)  # Implement this based on how you want to display results
    # response = predictions  # Implement this based on how you want to display results

    # print(data)
    # print(type(data))
    # print("\n\n\n\n\n")

    # response_data = {"prediction": cat_map[(response[0][0] > 0.5).astype("int")]}
    response = {"prediction": cat_map[(predictions[0][0] > 0.5).astype("int")]}
    # response = jsonify(response_data)
    # response = response_data
    # response = jsonify(type(response))
    # response = type(response)
    # response.headers()

    return response

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0') # for server
    # app.run(host='0.0.0.0', threaded=True, ssl_context=context) # for server
