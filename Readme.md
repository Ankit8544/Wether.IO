# **Weather.IO** ☁️

A weather forecasting application that provides real-time weather updates and predictive analytics for cities worldwide using the OpenWeatherMap API and TensorFlow for temperature predictions.

### **Live Link** 🔗
[Access the live app here](https://wether-io.onrender.com)

---

## **Table of Contents**

- [Project Overview](#project-overview)
- [Features](#features)
- [Data Processing & Model Training](#data-processing--model-training)
- [Application Architecture](#application-architecture)
- [Flask Integration & Deployment](#flask-integration--deployment)
- [API Usage & Key Management](#api-usage--key-management)
- [Usage Instructions](#usage-instructions)
- [Setup Guide](#setup-guide)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

---

## **Project Overview**

**Weather.IO** is a state-of-the-art weather forecasting application that delivers real-time weather updates for any city worldwide. It offers features such as temperature, humidity, wind speed, and pressure updates, along with predictive analytics for up to five days. The system leverages the OpenWeatherMap API for live data and uses machine learning (LSTM) for temperature predictions. Additionally, the app includes interactive visualizations and geolocation-based weather fetching for a seamless user experience.

---

## **Features**

- **Real-time Weather Data**: Provides live weather updates, including temperature, humidity, wind speed, and pressure.
- **7-Day Weather Forecast**: Detailed weather forecasts for the next 7 days.
- **Temperature Predictions**: Predicts temperature trends up to 5 days ahead using a machine learning model (LSTM).
- **Geolocation Detection**: Automatically detects the user's location and shows the relevant weather.
- **User Preferences**: Allows users to save favorite cities for quick access.
- **Interactive Visualizations**: Displays graphs for temperature trends using libraries like Matplotlib or Plotly.

---

## **Data Processing & Model Training**

### **Data Source**  
Weather data is collected using the **OpenWeatherMap API**, which provides real-time updates and historical weather data used for training the predictive models.

### **Machine Learning Model**
- The model is an **LSTM (Long Short-Term Memory)** neural network built with **TensorFlow** and **Keras** to forecast temperature trends based on historical weather data.
- Historical weather data is processed and fed into the LSTM model, which predicts temperature changes for up to 5 days ahead.
  
### **Model Storage**  
The trained LSTM model is stored as a **TensorFlow SavedModel** for easy deployment and real-time inference.

---

## **Application Architecture**

The app is built with a **microservices architecture**:

1. **Frontend**: Built using **HTML5**, **CSS3**, and **Bootstrap 5** for a responsive and clean user interface.
2. **Backend**: The Flask server handles API requests, manages user sessions, and serves the machine learning model for temperature predictions.
3. **Machine Learning Service**: The machine learning model is hosted as a separate service, which handles predictions.
4. **API Integration**: Fetches live weather data from the OpenWeatherMap API and uses TensorFlow for predictive analytics.

---

## **Flask Integration & Deployment**

### **Flask Setup**
The Flask web app serves the real-time weather data, processes user requests, and handles machine learning predictions. The two main routes are:

- `/`: Displays the weather details of the user's current or searched city.
- `/predict`: Provides a temperature prediction for the next 5 days using the LSTM model.

### **Deployment**
The app is deployed on **Render**, where Flask handles user interaction, API integration, and prediction services.

---

## **API Usage & Key Management**

The **OpenWeatherMap API** is used to fetch live weather data, and keys are managed through the `.env` file. The app reads the API key from the environment, and you can get your own API key from [OpenWeatherMap](https://openweathermap.org/api).

### **API Key Setup**
1. Create a `.env` file in the root directory.
2. Add your OpenWeatherMap API key:
   ```plaintext
   API_KEY=your_openweathermap_api_key
   ```

---

## **Usage Instructions**

1. **Homepage**:  
   - Shows the weather details based on your current geolocation.
   - You can also search for a city's weather information manually.
   
2. **Temperature Prediction**:  
   - Provides predictive temperature graphs using the LSTM model for up to 5 days in advance.

3. **Favorites**:  
   - Save your favorite cities for quick access to their weather data.

---

## **Setup Guide**

### **Requirements**
- Python 3.7+
- Flask
- TensorFlow, Keras
- Pandas, NumPy
- OpenWeatherMap API Key

### **Steps to Run Locally**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/weather-forecasting-app.git
   cd weather-forecasting-app
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the `.env` file with your OpenWeatherMap API key.

5. Run the Flask app:
   ```bash
   python app.py
   ```

6. Visit the app at `http://127.0.0.1:5000` in your browser.

---

## **Technologies Used**

- **Frontend**: HTML5, CSS3, Bootstrap 5
- **Backend**: Flask
- **Machine Learning**: TensorFlow, Keras
- **Data Handling**: Pandas, NumPy
- **API**: OpenWeatherMap API
- **Visualization**: Matplotlib, Plotly
- **Environment Management**: Python Dotenv for environment variables

---

## **Future Improvements**

1. **Enhanced Prediction Model**: Incorporating other weather features like humidity and wind speed into the machine learning model for more accurate predictions.
2. **User Authentication**: Implement OAuth2 for secure user authentication and personalized settings.
3. **Mobile Responsiveness**: Further UI optimizations for mobile and tablet devices.
4. **Weather Alerts**: Add push notifications for severe weather conditions based on user preferences.

---
