# Weather Forecasting Application

## Overview

The **Weather Forecasting Application** is a state-of-the-art web application built with Flask and TensorFlow. It delivers real-time weather updates and predictive analytics for cities worldwide, utilizing the OpenWeatherMap API for data acquisition and advanced machine learning models for temperature forecasting.

### Live Demo

Experience the application live at [this link](https://wether-io.onrender.com).

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Architecture](#architecture)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Testing](#testing)
- [Deployment](#deployment)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [License](#license)

## Features

- **Real-time Weather Data**: Instantaneous access to current weather parameters such as temperature, humidity, wind speed, and pressure.
- **7-Day Weather Forecast**: Comprehensive forecasts for the next week, including temperature ranges and weather conditions.
- **Temperature Predictions**: Utilizes an LSTM model to forecast temperature trends based on historical weather data for up to five days ahead.
- **Geolocation Detection**: Automatically fetches weather data based on the user's current location.
- **User Preferences**: Save favorite cities for quick access.
- **Interactive Visualizations**: Provides graphs for temperature trends and predictions using Matplotlib or Plotly.

## Technologies Used

- **Frontend**: HTML5, CSS3, Bootstrap 5 for responsive design
- **Backend**: Flask (Python) for RESTful APIs
- **Machine Learning**: TensorFlow, Keras for LSTM modeling
- **Data Handling**: Pandas, NumPy for data manipulation
- **API Integration**: OpenWeatherMap API for live weather data
- **Environment Management**: Python Dotenv for handling environment variables
- **Visualization**: Matplotlib, Plotly for dynamic data visualization
- **Testing**: Pytest for unit testing

## Architecture

The application follows a microservices architecture with the following components:

1. **Frontend**: User interface built with HTML/CSS and Bootstrap.
2. **Backend**: Flask server handles API requests and manages user sessions.
3. **Machine Learning Service**: Separate service that manages model inference and predictions using TensorFlow.
4. **Database**: (Optional) A relational database like PostgreSQL or a NoSQL database for storing user preferences and historical data.

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/weather-forecasting-app.git
   cd weather-forecasting-app
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your OpenWeatherMap API key:

   ```plaintext
   API_KEY=your_openweathermap_api_key
   ```

## Running the Application

1. Start the Flask server:

   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000`.

## Testing

To ensure the application functions as expected, run the test suite:

```bash
pytest tests/
```

### Test Coverage

The application includes tests for:

- API endpoints
- Data validation
- Model predictions

## Deployment

The application is deployed using Render, Heroku, or AWS Elastic Beanstalk. You can visit the live link to experience the features. Deployment scripts and configurations can be found in the `/deploy` directory.

## Usage

- **Home Page**: Automatically detects the user's location and displays relevant weather data.
- **Search Functionality**: Enter a city name to view its current weather and forecasts.
- **Temperature Predictions**: View predictive graphs for temperature trends.
- **Save Preferences**: Users can save favorite cities for quick access.

## Future Improvements

- **Enhanced Model Accuracy**: Incorporate additional features such as humidity and wind speed into the predictive model.
- **User Authentication**: Implement OAuth2 for secure user authentication and data privacy.
- **Mobile Responsiveness**: Optimize UI for better mobile experience.
- **Notification System**: Push notifications for severe weather alerts based on user preferences.

---