import requests

url = "http://shikib.com/pc_usr_data.json"
response = requests.get(url)

if response.status_code == 200:
    with open("persona_chat.json", "wb") as file:
        file.write(response.content)
        print("File downloaded successfully!")
else:
    print("Failed to download the file.")