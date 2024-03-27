# app.py

from flask import Flask, request, jsonify
from score import score
import joblib

# Construct the absolute path to the pickle file
pickle_file_path = 'best_text_classifier.pkl'

# Load the best model saved during experiments
model = joblib.load(pickle_file_path)

app = Flask(__name__)

@app.route('/score', methods=['POST'])
def get_score():
    data = request.json
    text = data['text']
    threshold = float(data['threshold'])
    
    # Score the text using the best model
    prediction, propensity = score(text, model, threshold)
    
    response = {
        'prediction': int(prediction),
        'propensity': propensity
    }
    print('Received Response')
    return jsonify(response)

if __name__ == '__main__':
    print('Running the app')
    app.run(debug=True, host='0.0.0.0')  # Run the Flask app; we should run using host 0.0.0.0 to make sure that docker can run it correctly
