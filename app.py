from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

model = joblib.load('best_regression_model.joblib')
# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    unemployment = float(request.form['unemployment'])
    mortgage = float(request.form['mortgage'])
    gdp = float(request.form['gdp'])
    foreclosures = float(request.form['foreclosures'])
    
    # Perform prediction using the loaded model
    input_data = [[unemployment, mortgage, gdp, foreclosures]]
    prediction = model.predict(input_data)
    
    # Render the prediction result template
    return render_template('result.html', prediction=prediction[0])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
