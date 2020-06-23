import json
import requests
url = "http://localhost:8000/inference" 
data = {"image_path" : 'IMG20180905151122.jpg'} 
data= json.dumps(data)
r = requests.post(url = url, data=data) 
out=r.text
print(out)