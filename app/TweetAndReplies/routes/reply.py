from typing import Optional,List
from fastapi import FastAPI, HTTPException, Depends, Request,status,Body,APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from TweetAndReplies.hashing import Hash
from TweetAndReplies.jwttoken import create_access_token
from TweetAndReplies.oauth import get_current_user
from datetime import datetime, timedelta
import TweetAndReplies.schemas as schemas
from bson import ObjectId
import json
import TweetAndReplies.main as main
router = APIRouter()
@router.post('/', status_code=status.HTTP_201_CREATED)
async def create_reply(reply: schemas.RepliesBaseSchema, userReturned: str = Depends(get_current_user)):
    reply=reply.__dict__
    reply.setdefault("user","")
    print(userReturned)
    user_id=userReturned.id
    user= main.User.find_one({"_id": ObjectId(user_id)})
    if user:
        print("user is:",user)
    tweet= main.Tweet.find_one({"_id":ObjectId(reply['tweet_id'])})
    rep= main.Replies.find_one({"_id":ObjectId(reply['tweet_id'])})
    if tweet==None and rep==None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail = f'No tweet found with this {reply.tweet_id} id')
   
    reply['user']= ObjectId(user_id)
    reply['created_at'] = datetime.utcnow()
    reply['updated_at'] = reply['created_at']
    reply['classification']={
        "prediction":reply['classification'].prediction,
        "Probability":reply['classification'].Probability
    }
    #classification(predict='Negative', Probability=0.27
    
    try:
        result = main.Replies.insert_one(reply)

        print("final results: ",result.inserted_id)
       
        new_reply =  main.Replies.find_one(result.inserted_id)
        print(new_reply)
        
        new_reply['_id']=str(new_reply['_id'])
        new_reply['user']=str(new_reply['user'])
        new_reply['tweet_id']=str(new_reply['tweet_id'])
        print("New reply : ",new_reply)
        return new_reply

    except:
      return {"Error":"Error in replying"}

def handleRecursion(replies:list):
    for rep in replies:
        for r in replies:
            if r['tweet_id']==str(rep['_id']):
                rep['replies'].append(r)

    for rep in replies:
        rep['_id']=str( rep['_id'])
        rep['user']=str(rep['user'])
        rep['created_at']=str(rep['created_at'])
        rep['updated_at']=str(rep['updated_at'])
        rep.setdefault('replies',[])
        if rep['replies']==[]:
            pass
        else:
            for repl in rep['replies']:
                if(repl!="string"):
                
                    repl['_id']=str( repl['_id'])
                    repl['user']=str(repl['user'])
                    repl['created_at']=str(repl['created_at'])
                    repl['updated_at']=str(repl['updated_at'])
        
def handleRecursiveReplies(replies:any):
  
     if(replies['replies'])==[]:
        pass
     else:
        for rep in replies['replies']:
            print(rep)
            print("Type>> ",type(rep))
            if(rep!="string"):
                rep['_id']=str( rep['_id'])
                rep['user']=str(rep['user'])
                rep['created_at']=str(rep['created_at'])
                rep['updated_at']=str(rep['updated_at'])
                rep.setdefault('replies',[])
                if rep['replies']==[]:
                    pass
                else:
                    for repl in rep['replies']:
                    
                        print(repl=="string")

                        if(repl!="string"):
                           
                            
                            repl['_id']=str( repl['_id'])
                        
                            repl['user']=str(repl['user'])
                            repl['created_at']=str(repl['created_at'])
                            repl['updated_at']=str(repl['updated_at'])
                            
                            print("Reply>><<<>>",repl)
                            print("\n")
                            if repl['replies']==[]:
                                pass
                            else:
                                handleRecursiveReplies(repl)


#Rearrange the tweets and replies again
def Rearrange_again(replies):

    postive_replies=[rep for rep in replies if rep.get('classification').get('prediction')=='Positive']
    negative_replies=[rep for rep in replies if rep.get('classification').get('prediction')=='Negative']
    neutral_replies=[rep for rep in replies if rep.get('classification').get('prediction')=='Neutral']

    postive_replies=sorted(postive_replies,key=lambda x:x.get('classification').get('Probability'),reverse=True)
  
    negative_replies=sorted(negative_replies,key=lambda x:x.get('classification').get('Probability'),reverse=False)
    neutral_replies=sorted(neutral_replies,key=lambda x:x.get('classification').get('Probability'),reverse=True)
    
    replies=postive_replies+neutral_replies+negative_replies
  
    return replies
@router.get('/',response_description="List of all replies")
async def get_replies(user: str = Depends(get_current_user)):
    replies= list(main.Replies.find())
    
    #print("found replies: ",replies)
    
    for rep in replies:
        for r in replies:
            if r['tweet_id']==str(rep['_id']):
                rep['replies'].append(r)

    for rep in replies:
        rep['_id']=str( rep['_id'])
        rep['user']=str(rep['user'])
        rep['created_at']=str(rep['created_at'])
        rep['updated_at']=str(rep['updated_at'])
        rep.setdefault('replies',[])
        # if rep['replies']==[]:
        #     pass
        # else:
        #     handleRecursiveReplies(rep)
        # print(rep)
        #rep=json.dumps(rep)
        print(rep)
        print(type(rep))
    # replies = await main.Replies.find().to_list(1000)
    # print(replies)
    replies=Rearrange_again(replies)
    print("Replies:: ",replies)
    return {'status': 'success', 'replies': replies}

@router.get(
    "/{id}", response_description="Get a single reply"
)
async def get_reply(id: str,user: str = Depends(get_current_user)):
    replies= list(main.Replies.find())
    print("User>> ",user)
    if(len(id)!=24):
        raise HTTPException(status_code=404, detail=f" reply id must have 24 length")
    
    try:
        id=ObjectId(id)
        if (reply :=  main.Replies.find_one({"_id":  id})) is not None:
            for rep in replies:
                
                if rep['tweet_id']==str(reply['_id']):
                    reply['replies'].append(rep)

            print("found reply:> ",reply)
            reply['_id']=str( reply['_id'])
            reply['user']=str(reply['user'])
            reply['created_at']=str(reply['created_at'])
            reply['updated_at']=str(reply['updated_at'])
            reply.setdefault('replies',[])
            if reply['replies']==[]:
                pass
            else:
                handleRecursiveReplies(reply)
            print(reply)
            reply=json.dumps(reply)
            print(reply)
            print(type(reply))
            return json.loads(reply)

        raise HTTPException(status_code=404, detail=f"reply {id} not found")

    except:
      raise HTTPException(status_code=404, detail=f"tweet {id} not found")

    
@router.put("/{id}", response_description="Update a reply", response_model=schemas.TweetBaseSchema)
async def update_reply(id: str, reply: schemas.UpdateTweetSchema = Body(...),user: str = Depends(get_current_user)):
    reply = {k: v for k, v in reply.dict().items() if v is not None}
    if(len(id)!=24):
        raise HTTPException(status_code=404, detail=f" reply id must have 24 length")
    
    if len(reply) >= 1:
        update_result =  main.Replies.update_one({"_id": ObjectId(id)}, {"$set": reply})

        if update_result.modified_count == 1:
            if (
                updated_reply:= main.Replies.find_one({"_id": id})
            ) is not None:
                return reply

    if (existing_reply := main.Replies.find_one({"_id": ObjectId(id)})) is not None:
        return existing_reply

    raise HTTPException(status_code=404, detail=f"reply {id} not found")


