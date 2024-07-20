import requests
TOKEN = ""
url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
print(requests.get(url).json())

import requests
TOKEN = ""
chat_id = ""
message = ""
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
print(requests.get(url).json())

