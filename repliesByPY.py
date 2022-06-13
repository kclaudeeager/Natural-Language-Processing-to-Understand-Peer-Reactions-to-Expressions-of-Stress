import json

with open('updatedJson.json',encoding='UTF-8') as jsonFile:
    print(jsonFile)
    data=json.load(jsonFile)
    print(data)
    