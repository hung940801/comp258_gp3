import pandas as pd

# Assuming the CSV file is in the same directory as your server script
data = pd.read_csv('./X_train.csv')

from flask import Flask, jsonify
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('./project_bestmodel.h5')

@app.route('/predict', methods=['GET'])
def predict():
    # Preprocess the data as needed
    processed_data = preprocess(data)  # Implement this function based on your model's needs

    # Make predictions
    predictions = model.predict(processed_data)

    # Convert predictions to a suitable format for JSON response
    response = format_predictions(predictions)  # Implement this based on how you want to display results

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
