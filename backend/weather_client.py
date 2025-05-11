import requests


class WeatherAPIClient:

    def __init__(self, token):
        self.token = token
        self.base_url = 'http://api.weatherapi.com/v1/current.json'
    
    def get_weather(self):
        
        try:
            response = requests.get(
                url=f'{self.base_url}?key={self.token}&q=Казань&aqi=no',
                timeout=5,
            ).json()

            return {
                'region': response['location']['name'],
                'temperature': response['current']['temp_c'],
                'wind': response['current']['wind_kph'],
            }
        except requests.RequestException:
            return None
        except KeyError:
            return None
