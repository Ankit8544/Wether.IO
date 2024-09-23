### IMPORTS - 
from flask import Flask, render_template, request
import requests
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

Metropolitan_Cities_Details = {
    "Patna": {
        "State/UT": "Bihar",
        "Latitude": 25.5941,
        "Longitude": 85.1376,
    },
    "National Capital Region": {
        "State/UT": "Delhi, Haryana, Rajasthan, Uttar Pradesh",
        "Latitude": 28.6139,
        "Longitude": 77.2090,
    },
    "Mumbai": {
        "State/UT": "Maharashtra",
        "Latitude": 19.0760,
        "Longitude": 72.8777,
    },
    "Kolkata": {
        "State/UT": "West Bengal",
        "Latitude": 22.5726,
        "Longitude": 88.3639,
    },
    "Chennai": {
        "State/UT": "Tamil Nadu",
        "Latitude": 13.0827,
        "Longitude": 80.2707,
    },
    "Bangalore": {
        "State/UT": "Karnataka",
        "Latitude": 12.9716,
        "Longitude": 77.5946,
    },
    "Hyderabad": {
        "State/UT": "Telangana",
        "Latitude": 17.3850,
        "Longitude": 78.4867,
    },
    "Pune": {
        "State/UT": "Maharashtra",
        "Latitude": 18.5204,
        "Longitude": 73.8567,
    },
    "Andhra Pradesh Capital Region": {
        "State/UT": "Andhra Pradesh",
        "Latitude": 16.5062,
        "Longitude": 80.6480,
    },
    "Ahmedabad": {
        "State/UT": "Gujarat",
        "Latitude": 23.0225,
        "Longitude": 72.5714,
    },
    "Surat": {
        "State/UT": "Gujarat",
        "Latitude": 21.1702,
        "Longitude": 72.8311,
    },
    "Visakhapatnam": {
        "State/UT": "Andhra Pradesh",
        "Latitude": 17.6868,
        "Longitude": 83.2185,
    },
    "Jaipur": {
        "State/UT": "Rajasthan",
        "Latitude": 26.9124,
        "Longitude": 75.7873,
    },
    "Lucknow": {
        "State/UT": "Uttar Pradesh",
        "Latitude": 26.8467,
        "Longitude": 80.9462,
    },
    "Kanpur": {
        "State/UT": "Uttar Pradesh",
        "Latitude": 26.4499,
        "Longitude": 80.3319,
    }
}
DEFAULT_CITY_NAME = 'Patna'
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

def get_current_weather():
    latitude, longitude = User_Current_Location()
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

def get_address_nominatim():
    latitude, longitude = User_Current_Location()
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

def fetch_current_weather_forecast():
    num_days = 5
    latitude, longitude = User_Current_Location()
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
def load_model_and_scaler(city_name):
    model = load_model(f"./artifacts/{city_name}_model.h5")#{city_name}_model.h5")
    with open(f"./artifacts/{city_name}_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Function to make predictions
def predict_temperature(city_name):
    model, scaler = load_model_and_scaler(city_name)
    
    # Prepare the last 7 days of test data
    last_7_days = np.random.rand(7, 3)  # Replace with actual last 7 days data
    last_7_days_scaled = scaler.transform(last_7_days)
    
    predictions = []
    input_sequence = last_7_days_scaled.copy()

    for _ in range(5):  # Predict for the next 5 days
        pred = model.predict(np.array([input_sequence]))
        predictions.append(pred[0][0])
        
        pred_reshaped = np.zeros((1, input_sequence.shape[1]))
        pred_reshaped[0, 0] = pred
        
        input_sequence = np.vstack((input_sequence[1:], pred_reshaped))
    
    # Inverse transform predictions back to original scale
    predictions = scaler.inverse_transform([[pred, 0, 0] for pred in predictions])[:, 0]
    
    predicted_dates = [(datetime.now().date() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(5)]
    return {
        'Date': predicted_dates,
        'Predicted Temperature': predictions
    }


app = Flask(__name__)

@app.route('/')
def index():
    latitude, longitude = User_Current_Location()
    location = get_address_nominatim() if latitude and longitude else None
    current_weather = get_current_weather() if latitude and longitude else None
    current_location_forecast = fetch_current_weather_forecast() if latitude and longitude else []

    return render_template('index.html',
                           latitude=latitude,
                           longitude=longitude,
                           location=location,
                           current_weather=current_weather,
                           current_location_forecast=current_location_forecast)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    CITY_NAME = request.args.get('city','Patna')
    result = predict_temperature(city_name=CITY_NAME)
    result['Predicted Temperature'] = [round(temp, 2) for temp in result['Predicted Temperature']]
    day_names = [datetime.strptime(date_str, '%Y-%m-%d').strftime('%a') for date_str in result['Date']]
    combined_data = zip(day_names, result['Predicted Temperature'])
    return render_template('report.html', dates=day_names, temperatures=result['Predicted Temperature'], result=combined_data, city_name=CITY_NAME)
    
if __name__ == '__main__':
    app.run()
