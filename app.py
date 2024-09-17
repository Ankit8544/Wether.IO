### IMPORTS - 
from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import mysql.connector
import matplotlib.pyplot as plt
import io
import random
import geocoder
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
API_KEY = os.getenv("API_KEY")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
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
City_List = ['Patna', 'National Capital Region', 'Mumbai', 'Kolkata', 'Chennai', 'Bangalore', 'Hyderabad', 'Pune', 'Andhra Pradesh Capital Region', 'Ahmedabad', 'Surat', 'Visakhapatnam', 'Jaipur', 'Lucknow', 'Kanpur']
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
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={API_KEY}&units=metric'
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
    forecast_url = f"https://api.openweathermap.org/data/3.0/onecall?lat={latitude}&lon={longitude}&exclude=minutely,hourly,alerts&appid={API_KEY}&units=metric"
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
def  connect_to_db():
    # Create a connection to the database
    conn = mysql.connector.connect(
        host = MYSQL_HOST,
        port=3306,
        user = MYSQL_USER,
        password = MYSQL_PASSWORD,
        database = MYSQL_DATABASE
    )
    return conn  

def check_data(city_name):
    conn = connect_to_db()
    cursor = conn.cursor()
    # Check if the table exists in the database
    query = f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'weather_data' AND table_name = '{city_name}'"
    cursor.execute(query)
    table_exists = cursor.fetchone()[0]
    # Initialize df to handle cases where the table does not exist
    df = None
    # If the table exists, load it into a DataFrame
    if table_exists:
        df = pd.read_sql(f"SELECT * FROM {city_name}", conn)
    else:
        print(f"Table '{city_name}' does not exist in the database.")
    # Close the connection
    cursor.close()
    conn.close()
    return df

def fetch_weather_data(city, start_date, end_date):
    LAT = Metropolitan_Cities_Details[city]['Latitude']
    LON = Metropolitan_Cities_Details[city]['Longitude']
    weather_data = {
        'date': [],
        'temp_min': [],
        'temp_max': [],
        'humidity': [],
        'wind_speed': []
    }
    # Loop through the dates
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')    
        # Make the API call for each date
        url = f"{BASE_URL}?lat={LAT}&lon={LON}&date={date_str}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            min_temp = data['temperature']['min']
            max_temp = data['temperature']['max']
            humidity = data['humidity']['afternoon']
            wind_speed = data['wind']['max']['speed']                
            weather_data['date'].append(current_date)
            weather_data['temp_min'].append(min_temp)
            weather_data['temp_max'].append(max_temp)
            weather_data['humidity'].append(humidity)
            weather_data['wind_speed'].append(wind_speed)
        # Move to the next date
        current_date += timedelta(days=1)
    # Create a DataFrame from the collected data
    df = pd.DataFrame(weather_data)
    return df

def preprocess_data(df):
    # Sort DataFrame by date
    df = df.sort_values('date')
    
    # Compute daily average temperature
    df['temp_avg'] = (df['temp_min'] + df['temp_max']) / 2
    
    # Define features and target variable
    features = ['temp_avg', 'humidity', 'wind_speed']
    target = 'temp_avg'
    
    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features])
    
    # Prepare X and y
    X = []
    y = []
    
    # Use the previous 7 days to predict the next day's temperature
    for i in range(7, len(scaled_data)):
        X.append(scaled_data[i-7:i])
        y.append(scaled_data[i][0])  # temp_avg is the target variable
    
    # Convert lists to numpy arrays
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_next_days(model, last_7_days, scaler):
    predictions = []
    input_sequence = last_7_days.copy()
    
    for _ in range(5):  # Predict for the next 5 days
        # Predict the next day
        pred = model.predict(np.array([input_sequence]))  # Shape (1, 1)
        predictions.append(pred[0][0])
        
        # Reshape pred to have the same number of features as input_sequence
        pred_reshaped = np.zeros((1, input_sequence.shape[1]))  # Create a zero array with correct shape
        pred_reshaped[0, 0] = pred  # Assign the prediction to the correct position
        
        # Shift the input sequence forward, appending the new prediction
        input_sequence = np.vstack((input_sequence[1:], pred_reshaped))
    
    # Inverse transform predictions back to original scale
    predictions = scaler.inverse_transform([[pred, 0, 0] for pred in predictions])[:, 0]
    
    return predictions 

def save_to_mysql(df, city_name):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()

        # Create table query (using city name as table name)
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS `{city_name}` (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE,
            temp_min FLOAT,
            temp_max FLOAT,
            humidity FLOAT,
            wind_speed FLOAT
        );
        """
        cursor.execute(create_table_query)
        
        # Insert data into the table
        for index, row in df.iterrows():
            insert_query = f"""
            INSERT INTO `{city_name}` (date, temp_min, temp_max, humidity, wind_speed)
            VALUES (%s, %s, %s, %s, %s);
            """
            cursor.execute(insert_query, (row['date'], row['temp_min'], row['temp_max'], row['humidity'], row['wind_speed']))
        
        # Commit the changes
        conn.commit()
        print(f"Data successfully saved to {city_name} table in MySQL.")
    
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def append_new_to_mysql(df, city_name):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()

        # Fetch existing dates from the table
        cursor.execute(f"SELECT date FROM `{city_name}`")
        existing_dates = set(row[0] for row in cursor.fetchall())

        # Filter out rows in df that already exist in the MySQL table
        df['date'] = pd.to_datetime(df['date'])  # Ensure 'date' column is in datetime format
        df_to_insert = df[~df['date'].isin(existing_dates)]

        # Insert only new data into the table
        if not df_to_insert.empty:
            insert_query = f"""
            INSERT INTO `{city_name}` (date, temp_min, temp_max, humidity, wind_speed)
            VALUES (%s, %s, %s, %s, %s);
            """
            for index, row in df_to_insert.iterrows():
                cursor.execute(insert_query, (row['date'].strftime('%Y-%m-%d'), row['temp_min'], row['temp_max'], row['humidity'], row['wind_speed']))

            # Commit the changes
            conn.commit()
            print(f"New data successfully appended to {city_name} table in MySQL.")
        else:
            print("No new data to append.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def main(CITY_NAME, City_List, Start_Date, Current_Date):
    # Ensure Current_Date is a date object
    if isinstance(Current_Date, datetime):
        Current_Date = Current_Date.date()

    if CITY_NAME in City_List:
        Available_df = check_data(CITY_NAME.lower())
        
        if Available_df is not None and not Available_df.empty:
            
            last_date = Available_df.tail(1)['date'].values[0]
            if isinstance(last_date, datetime):
                last_date = last_date.date()
            
            if Current_Date == last_date:
                    
                X, y, scaler = preprocess_data(Available_df)
                X_train, X_test = X[:-5], X[-5:]
                y_train, y_test = y[:-5], y[-5:]
                model = build_model((X_train.shape[1], X_train.shape[2]))
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[early_stopping], verbose=0)
                last_7_days = X_test[-1]
                predicted_temperatures = predict_next_days(model, last_7_days, scaler)
                predicted_dates = [(Current_Date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(5)]
                Final_Outcome = {
                    'Date': predicted_dates,
                    'Predicted Temperature': predicted_temperatures
                }     
            
            else:
                
                New_Start_Date = last_date + timedelta(days=1)
                New_df = fetch_weather_data(CITY_NAME, New_Start_Date, Current_Date)
                Updated_df = pd.concat([Available_df, New_df], ignore_index=True)
                Updated_df = Updated_df.drop_duplicates(subset='date', keep='last')
                append_new_to_mysql(Updated_df, CITY_NAME.lower())
                X, y, scaler = preprocess_data(Updated_df)
                X_train, X_test = X[:-5], X[-5:]
                y_train, y_test = y[:-5], y[-5:]
                model = build_model((X_train.shape[1], X_train.shape[2]))
                early_stopping = EarlyStopping( monitor='val_loss', patience=10, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[early_stopping], verbose=0)
                last_7_days = X_test[-1]
                predicted_temperatures = predict_next_days(model, last_7_days, scaler)
                predicted_dates = [(Current_Date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(5)]
                Final_Outcome = {
                    'Date': predicted_dates,
                    'Predicted Temperature': predicted_temperatures
                }
                    
        else:
            
            New_df = fetch_weather_data(CITY_NAME, Start_Date, Current_Date)
            save_to_mysql(New_df, CITY_NAME.lower())
            X, y, scaler = preprocess_data(New_df)
            X_train, X_test = X[:-5], X[-5:]
            y_train, y_test = y[:-5], y[-5:]
            model = build_model((X_train.shape[1], X_train.shape[2]))
            early_stopping = EarlyStopping( monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[early_stopping])
            last_7_days = X_test[-1]
            predicted_temperatures = predict_next_days(model, last_7_days, scaler)
            predicted_dates = [(Current_Date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(5)]
            Final_Outcome = {
                'Date': predicted_dates,
                'Predicted Temperature': predicted_temperatures
            }

        return Final_Outcome
    else :
        return f"We are not able to forecast the weather condition of {CITY_NAME}"

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
    CITY_NAME = request.args.get('city')
    result = main(CITY_NAME=DEFAULT_CITY_NAME, City_List=City_List, Start_Date=Start_Date, Current_Date=Current_Date)
    result['Predicted Temperature'] = [round(temp, 2) for temp in result['Predicted Temperature']]
    day_names = [datetime.strptime(date_str, '%Y-%m-%d').strftime('%a') for date_str in result['Date']]
    combined_data = zip(day_names, result['Predicted Temperature'])
    return render_template('report.html', dates=day_names, temperatures=result['Predicted Temperature'], result=combined_data, city_name=CITY_NAME)

# Add this function to disable caching during development
@app.after_request
def add_header(response):
    # Disable caching during development
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response
    
if __name__ == '__main__':
    app.run()
