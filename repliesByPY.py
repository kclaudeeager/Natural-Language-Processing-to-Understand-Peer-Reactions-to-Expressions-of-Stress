import json
data={}
with open('updatedJson.json',encoding='UTF-8') as jsonFile:
    print(jsonFile)
    data=json.load(jsonFile)
    #print(len(data['data']))
# print(data.keys)
newDict=dict()
objectList=list(dict())

#print(type(data['includes']))
for (key, value),(key1,value1) in zip((data['includes']).items(),(data['data']).items()):
    value.setdefault('tweets',[])
    # print(key,value['tweets'])
    #print(key,len(value['tweets']))
    if len(value['tweets'])==0:
        #print(key,'No tweet found!')
        newDict['tweet']='No tweet found!'
        continue
        
    #print(key,value['tweets'][0]['text'])
    tweet=value['tweets'][0]['text']
    newDict.update({"tweet":value['tweets'][0]['text']})
    
    #print(tweet)
    replies=list()
    for reply in value1:
        replies.append(reply['text'])
    newDict.update({'replies':replies})
    objectList.append(newDict)
    #print(objectList)
    
print(objectList)
# for i in range(0,len(data['data'])):
#     newDict['tweet']=data['includes'][str(i)]['tweets'][0]['text']
# print(newDict)
        
    
    
    