# Weather Forecasting and Prediction App

This project is a Flask-based web application that fetches real-time weather data, provides weather forecasts, and predicts future weather conditions using machine learning (LSTM). The app uses the OpenWeather API to retrieve weather information for various cities and allows the user to explore current and historical weather data, and even predict future temperatures.

## Live Link
You can access the live version of this weather forecasting application [here](https://wether-io.onrender.com/).

## Features

- **Real-Time Weather**: Fetches current weather information based on the user's location or from selected metropolitan cities in India.
- **5-Day Weather Forecast**: Provides a detailed 5-day weather forecast, including temperature, humidity, wind speed, and weather description.
- **Historical Weather Data**: Fetches historical weather data for metropolitan cities, spanning multiple years.
- **Temperature Prediction**: Uses LSTM (Long Short-Term Memory) neural networks to predict future temperatures based on historical data.
- **Database Integration**: Weather data can be saved to a MySQL database and fetched back for analysis.
  
## Tech Stack

- **Frontend**: HTML, CSS, Flask templates (Jinja2)
- **Backend**: Flask, Python
- **APIs**: 
  - OpenWeather API for current weather and forecasts
  - IP-API for determining user's geolocation
  - Nominatim OpenStreetMap API for reverse geolocation
- **Machine Learning**: LSTM model for weather prediction
- **Database**: MySQL for saving historical weather data
- **Other Libraries**: 
  - `pandas`, `numpy` for data manipulation
  - `tensorflow.keras` for LSTM modeling
  - `matplotlib` for plotting
  - `sklearn.preprocessing` for data normalization

## Setup Instructions

### Prerequisites

1. Python 3.8+
2. Flask
3. MySQL Server
4. OpenWeather API Key

### Libraries

Install required libraries using the following command:

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory and add the following variables:

```bash
API_KEY=<Your_OpenWeather_API_Key>
MYSQL_HOST=<Your_MySQL_Host>
MYSQL_USER=<Your_MySQL_User>
MYSQL_PASSWORD=<Your_MySQL_Password>
MYSQL_DATABASE=<Your_Database_Name>
```

### Database Setup

Create a MySQL database, and ensure the `MYSQL_DATABASE` environment variable matches its name. The app will automatically create the required tables.

### Running the App

To start the Flask application, use the following command:

```bash
flask run
```

### API Endpoints

- **Current Weather**: Fetches the real-time weather for the user's current location or selected city.
- **Weather Forecast**: Fetches a 5-day weather forecast for a given city.
- **Historical Weather**: Retrieves historical weather data from the OpenWeather API for specified dates and cities.
- **Prediction**: Predicts future temperatures for the next 5 days using an LSTM model.

## Key Functions

### 1. `User_Current_Location()`
Fetches the user's current latitude and longitude based on their IP address.

### 2. `get_current_weather()`
Uses the OpenWeather API to get the current weather based on the user's location.

### 3. `fetch_current_weather_forecast()`
Retrieves a 5-day weather forecast from the OpenWeather API.

### 4. `fetch_weather_data(city, start_date, end_date)`
Fetches historical weather data for a given city and date range.

### 5. `preprocess_data(df)`
Prepares the data for model training by normalizing it and extracting features for prediction.

### 6. `build_model(input_shape)`
Builds the LSTM model to predict future weather temperatures.

### 7. `predict_next_days(model, last_7_days, scaler)`
Predicts the next 5 days' weather using the trained LSTM model.

### 8. `save_to_mysql(df, city_name)`
Saves weather data to a MySQL database.

## Data Flow

1. User accesses the app to get current weather or forecast.
2. For historical data or predictions, the app retrieves and processes historical weather data.
3. The user can view real-time, past, or predicted weather data via the web interface.

## Potential Enhancements

- **Additional Cities**: Add more cities or allow users to input any city name for weather data retrieval.
- **Weather Visualizations**: Add more detailed visualizations (graphs, charts) for better data understanding.
- **Advanced Prediction Models**: Experiment with more advanced machine learning models or architectures for better predictions.
- **User Authentication**: Add user authentication for customized experiences (e.g., saving favorite cities).

## Troubleshooting

- Ensure your MySQL server is running and correctly configured.
- Check your API key in the `.env` file if weather data isn’t being fetched.
- Make sure the required libraries are installed via `pip install -r requirements.txt`.

## Author

Ankit Kumar