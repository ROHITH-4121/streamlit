import streamlit as st
import requests

API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"

API_KEY = "your_actual_api_key_here"
city = "London"
url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
response = requests.get(url)
print(response.status_code)
print(response.json())

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

st.title("🌤️ Weather Forecast Dashboard")

city = st.text_input("Enter city name", "London")

if city:
    data = get_weather(city)
    if data:
        st.subheader(f"Current Weather in {city.title()}")
        temp = data['main']['temp']
        desc = data['weather'][0]['description']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']

        st.write(f"Temperature: {temp} °C")
        st.write(f"Weather: {desc.title()}")
        st.write(f"Humidity: {humidity}%")
        st.write(f"Wind Speed: {wind_speed} m/s")
    else:
        st.error("City not found or API error.")

