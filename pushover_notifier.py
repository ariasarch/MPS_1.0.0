import requests
import time

class PushoverNotifier:
    def __init__(self, api_token: str, user_key: str):
        self.api_token = api_token
        self.user_key = user_key
        self.api_url = "https://api.pushover.net/1/messages.json"
        
    def send_notification(self, message: str, title: str) -> bool:
        payload = {
            "token": self.api_token,
            "user": self.user_key,
            "message": message,
            "title": title
        }
        
        try:
            response = requests.post(self.api_url, data=payload)
            response.raise_for_status()
            print(f"Notification sent: {title}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Failed to send notification: {str(e)}")
            return False