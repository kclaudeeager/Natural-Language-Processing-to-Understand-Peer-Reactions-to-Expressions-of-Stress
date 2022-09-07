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

@router.get('/',response_description="List of all replies")
async def get_replies(user: str = Depends(get_current_user)):
    replies= list(main.Replies.find())
    
    print("found replies: ")
    for rep in replies:
        for r in replies:
            r['_id']=str( r['_id'])
            r['user']=str(r['user'])
            r['created_at']=str(r['created_at'])
            r['updated_at']=str(r['updated_at'])
            r.setdefault('replies',[])
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
                repl['_id']=str( repl['_id'])
                repl['user']=str(repl['user'])
                repl['created_at']=str(repl['created_at'])
                repl['updated_at']=str(repl['updated_at'])
                
        print(rep)
        rep=json.dumps(rep)
        print(rep)
        print(type(rep))
    # replies = await main.Replies.find().to_list(1000)
    # print(replies)
    
    return {'status': 'success', 'replies': replies}


