from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_features)[0]

        return render_template('index.html',
                               prediction_text=f'🏠 Predicted House Price (MEDV): ${prediction:,.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
