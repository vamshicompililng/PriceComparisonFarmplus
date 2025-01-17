import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
# from flask import jsonify
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and encoders
model = None
label_encoders = {}

# Train and save the model if not already saved
def train_and_save_model():
    global model, label_encoders

    # Path to dataset
    dataset_path = 'dataset.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset not found. Please place the dataset.csv file in the same directory.")

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Encode categorical columns
    categorical_columns = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Define features and target variable
    X = df[['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']]
    y = df['Modal Price']

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model and encoders to disk
    joblib.dump(model, 'crop_price_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')

    # Evaluate model (Optional, for logging)
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R-squared:", r2_score(y_test, y_pred))
    print("Model and label encoders saved successfully!")

# Load the trained model and label encoders
def load_model_and_encoders():
    global model, label_encoders
    if os.path.exists('crop_price_model.pkl') and os.path.exists('label_encoders.pkl'):
        model = joblib.load('crop_price_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
    else:
        train_and_save_model()

# Define route for the main page
@app.route('/')
def home():
    return render_template('crop_pred_sorted.html')
@app.route('/predict', methods=['POST'])
def predict():
    predictions = []
    crop_names = []
    crop_prices = []
    for i in range(1, 6):
        try:
            State = request.form[f'State{i}']
            District = request.form[f'District{i}']
            Market = request.form[f'Market{i}']
            Commodity = request.form[f'Commodity{i}']
            Variety = request.form[f'Variety{i}']
            Grade = request.form[f'Grade{i}']

            # Prepare the input data for this crop
            input_data = pd.DataFrame([[State, District, Market, Commodity, Variety, Grade]],
                                      columns=['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade'])

            # Encode categorical features
            for column in ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']:
                input_data[column] = label_encoders[column].transform(input_data[column])

            # Make a prediction for this crop
            predicted_price = model.predict(input_data)[0]
            predicted_price_int = int(predicted_price)

            # Append the prediction and crop name for this crop
            predictions.append(f"Crop {i}: {predicted_price_int} Rupees Per Quintal")
            crop_names.append(Commodity)
            crop_prices.append(predicted_price_int)
        except KeyError:
            return render_template('crop_pred_sorted.html', prediction=" Invalid input: Some values are not recognized.", crop_names=[], crop_prices=[])

    # Join all predictions into a single string
    prediction_results = "<br>".join(predictions)

    # Return the predictions, crop names, and crop prices to the HTML page
    return render_template('crop_pred_sorted.html', prediction=prediction_results, crop_names=crop_names, crop_prices=crop_prices)

# ...existing code...

if __name__ == '__main__':
    # Load or train model and encoders
    load_model_and_encoders()

    # Run the Flask app
    app.run(debug=True)

