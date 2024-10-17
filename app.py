import numpy as np
import pickle
from flask import Flask, request, render_template

# Define a Flask app
app = Flask(__name__)

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print('Model loaded. Start serving...')
print('Check http://127.0.0.1:5000/')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/', methods=['POST'])
def get_data():
    val = ""
    if request.method == 'POST':
        # Get the input data from the form
        pregnancies = float(request.form['Pregnancies'])
        glucose = float(request.form['Glucose'])
        blood_pressure = float(request.form['BloodPressure'])
        skin_thickness = float(request.form['SkinThickness'])
        insulin = float(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        diabetes_pedigree = float(request.form['DiabetesPedigreeFunction'])
        age = float(request.form['Age'])
        
        # Create a 2D array for model input
        newpat = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age]])
        
        # Make the prediction
        result = model.predict(newpat)

        # Interpret the result
        val = "Diabetes" if result[0] == 1 else "No Diabetes"

    return render_template('index.html', value=val)

if __name__ == '__main__':
    app.run(debug=True)