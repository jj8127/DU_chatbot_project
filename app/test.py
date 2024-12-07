import requests

url = "http://127.0.0.1:8000/rag-query"
data = {"question": "동서울대학교 컴퓨터정보과에 대해서 알려줘?"}

response = requests.post(url, json=data)
print(response.json())
