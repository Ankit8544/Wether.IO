### IMPORTS - 
from flask import Flask, render_template, request
import requests
import numpy as np
from keras.models import load_model
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

BASE_URL = "https://api.openweathermap.org/data/3.0/onecall/day_summary"
Current_Date = datetime.now().date()
Start_Date = (datetime.now() - timedelta(days=3*365)).date()

### CURRENT LOCATION DETAIL - 
def User_Current_Location():
    try:
        response = requests.get('http://ip-api.com/json/', timeout=5)
        data = response.json()
        
        if response.status_code == 200:
            latitude = data['lat']
            longitude = data['lon']
            return latitude, longitude
        else:
            print(f"Error {response.status_code}: {data.get('message', 'Unable to retrieve location.')}")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None, None

def get_current_weather(latitude, longitude):
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid=6b33a74b8dd94a967b2622a5eb6c1d93&units=metric'
    response = requests.get(url)
    data = response.json()

    current_condition = {
        'Current Temperature': data['main']['temp'],
        'Humidity': data['main']['humidity'],
        'Pressure': data['main']['pressure'],
        'Visibility': data['visibility'] / 1000,
        'Wind Speed': round(data['wind']['speed'] * 3.6, 1),
        'Weather': [data['weather'][0]['description']][0]
    } 
    return current_condition

def get_address_nominatim(latitude, longitude):
    url = f"https://nominatim.openstreetmap.org/reverse?lat={latitude}&lon={longitude}&format=json"
    headers = {
        'User-Agent': 'WeatherForecastingAPP/1.0 (ankitkumar875740l@example.com)'  # Replace with your app name and email
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    
    address = data.get('address', {})
    
    # Safely extract information
    return {
        'Full Address': data.get('display_name', 'Unknown'),
        'City': address.get('city', 'Unknown'),
        'State District': address.get('state_district', 'Unknown'),
        'State': address.get('state', 'Unknown'),
        'Country': address.get('country', 'Unknown')
    }

def fetch_current_weather_forecast(latitude, longitude):
    num_days = 5
    # 7-Day Forecast API (One Call API)
    forecast_url = f"https://api.openweathermap.org/data/3.0/onecall?lat={latitude}&lon={longitude}&exclude=minutely,hourly,alerts&appid=6b33a74b8dd94a967b2622a5eb6c1d93&units=metric"
    forecast_response = requests.get(forecast_url)
    forecast_data = forecast_response.json()

    # Extract relevant forecast data starting from tomorrow
    forecast_list = []
    for day in forecast_data['daily'][1:num_days+1]:  # Fetch data for num_days
        date = datetime.fromtimestamp(day['dt'])
        formatted_date = date.strftime('%Y-%m-%d')
        day_name = date.strftime('%A')
        forecast = {
            "Date": formatted_date,
            "Day Name": day_name,
            "Temperature (°C)": round((day['temp']['day'] + day['temp']['night']) / 2, 2),
            "Humidity (%)": day['humidity'],
            "Weather Description": day['weather'][0]['description'],
            "Wind Speed (Km/h)": round(day['wind_speed'] * 3.6, 1)
        }
        forecast_list.append(forecast)

    return forecast_list

### METO CITY DETAIL -
# Function to load the model and scaler
def load_model_and_scaler_X_test(city_name):
    # Load the scaler
    with open(os.path.join(f"./artifacts/{city_name}_scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)

    # Load the last 7 days of test data
    with open(os.path.join(f"./artifacts/{city_name}_X_test_last_7_days.pkl"), 'rb') as f:
        X_test_last_7_days = pickle.load(f)
        
    model = load_model(f"./artifacts/{city_name}_model.h5")
    
    return model, scaler, X_test_last_7_days

# Function to make predictions
def predict_temperature(city_name):
    model, scaler, X_test_last_7_days = load_model_and_scaler_X_test(city_name)
    
    # Prepare to hold predictions
    predictions = []
    predicted_scaled_input = X_test_last_7_days[0]  # Start with the last input used for prediction

    # Predict for the next 5 days
    for _ in range(5):
        # Make a prediction
        predicted_scaled_temp_avg = model.predict(predicted_scaled_input.reshape(1, 7, 3))
        
        # Prepare for inverse transform
        # Create an array with the same shape as the original feature space
        pred_input = np.zeros((1, 3))
        pred_input[0, 0] = predicted_scaled_temp_avg.flatten()[0]  # Fill with predicted value
        # Fill humidity and wind speed with the last values or some sensible defaults
        pred_input[0, 1] = X_test_last_7_days[0][-1][1]  # Last humidity value
        pred_input[0, 2] = X_test_last_7_days[0][-1][2]  # Last wind speed value

        # Inverse transform to get actual temperature value
        predicted_temp_avg = scaler.inverse_transform(pred_input)
        
        # Round the predicted temperature to two decimals
        rounded_temp = round(predicted_temp_avg[0][0], 2)
        predictions.append(rounded_temp)  # Store the rounded predicted temperature

        # Update the input for the next prediction
        predicted_scaled_input = np.append(predicted_scaled_input[1:], pred_input, axis=0)

    # Get the current date and calculate the next 5 dates
    today = datetime.now()
    predicted_days = [(today + timedelta(days=i)).strftime('%A') for i in range(1, 6)]

    # Create the dictionary output
    result_dict = {
        'Days': predicted_days,
        'Predicted Temperature': np.array(predictions)  # Store predictions as an array
    }
    
    return result_dict

app = Flask(__name__)

@app.route('/')
def index():
    # latitude, longitude = User_Current_Location()
    latitude = 25.5941
    longitude = 85.1376
    location = get_address_nominatim(latitude, longitude)
    current_weather = get_current_weather(latitude, longitude)
    current_location_forecast = fetch_current_weather_forecast(latitude, longitude)

    return render_template('index.html',
                           latitude=latitude,
                           longitude=longitude,
                           location=location,
                           current_weather=current_weather,
                           current_location_forecast=current_location_forecast)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    CITY_NAME = request.args.get('city', 'Patna')
    result = predict_temperature(city_name=CITY_NAME)
    # Convert predicted temperatures to a list
    result['Predicted Temperature'] = result['Predicted Temperature'].tolist()
    day_names = result['Days']
    combined_data = zip(day_names, result['Predicted Temperature'])
    return render_template('report.html', dates=day_names, temperatures=result['Predicted Temperature'], result=combined_data, city_name=CITY_NAME)
  
if __name__ == '__main__':
    app.run()
