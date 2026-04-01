import requests

url = "https://raw.githubusercontent.com/naijilnj/sitest/master/filenames.py"

response = requests.get(url)

print(response.text)