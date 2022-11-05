from typing import Optional
from TweetAndReplies.routes.reply import Rearrange_again
from fastapi import FastAPI, HTTPException, Depends, Request,status,Body,APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from TweetAndReplies.hashing import Hash
from TweetAndReplies.jwttoken import create_access_token
from TweetAndReplies.oauth import get_current_user
from datetime import datetime, timedelta
import TweetAndReplies.schemas as schemas
import json
from sys import getsizeof
from bson import ObjectId
import TweetAndReplies.main as main
router = APIRouter()
@router.post('/', status_code=status.HTTP_201_CREATED)
async def create_tweet(tweet: schemas.TweetBaseSchema, userReturned: str = Depends(get_current_user)):
    
    tweet=tweet.__dict__
    tweet.setdefault("user","")
    print(userReturned)
    user_id=userReturned.id
    user=main.User.find_one({"_id": ObjectId(user_id)})
    if user:
        print("user is:",user)
    tweet['user']= ObjectId(user_id)
    tweet['created_at'] = datetime.utcnow()
    tweet['updated_at'] = tweet['created_at']
    
    #print(tweet)
    print("Attachements>>",tweet['attachements'])
    
    try:
        result =main.Tweet.insert_one(tweet)
        print("final results: ",result.inserted_id)
        # pipeline = [
        #     {'$match': {'_id': result.inserted_id}},
        #     {'$lookup': {'from': 'users', 'localField': 'user',
        #                  'foreignField': '_id', 'as': 'user'}},
        #     {'$unwind': '$user'},
        # ]
        new_tweet = main.Tweet.find_one(result.inserted_id)
        print(new_tweet)
        new_tweet['_id']=str(new_tweet['_id'])
        new_tweet['user']=str(new_tweet['user'])
        print("New tweet : ",new_tweet)
        return new_tweet
    except:
      return {"Error":"Error in tweeting"}

@router.get('/')
def get_tweets(user: str = Depends(get_current_user)):
   
    tweets = list(main.Tweet.find())
    replies= list(main.Replies.find())
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
        
                
    for tweet in tweets:
        print(type(tweet))
        print(tweet)
        tweet['_id']=str( tweet['_id'])
        tweet['user']=str(tweet['user'])
        tweet['created_at']=str(tweet['created_at'])
        tweet['updated_at']=str(tweet['updated_at'])
        tweet.setdefault('replies',[])
        replies=Rearrange_again(replies)
        for rep in replies:
            if rep['tweet_id']==tweet['_id']:
                tweet['replies'].append(rep)
        tweet=json.dumps(tweet)
             
    return {'status': 'success', 'tweets': tweets}
def get_reply(id: str,user: str = Depends(get_current_user)):
    replies= list(main.Replies.find())
    try:
        
        reply=main.Replies.find_one({"_id":  id})
        #print("reply ",reply)
        if reply is not None:
            for rep in replies:
                
                if rep['tweet_id']==str(reply['_id']):
                    rep['_id']=str(rep['_id'])
                    
                    rep['user']=str(rep['user'])
                    rep['created_at']=str(rep['created_at'])
                    rep['updated_at']=str(rep['updated_at'])
                    reply['replies'].append(rep)

            #print("found reply:> ",reply)
            reply['_id']=str( reply['_id'])
            reply['user']=str(reply['user'])
            reply['created_at']=str(reply['created_at'])
            reply['updated_at']=str(reply['updated_at'])
    
            print("Replies so far",reply)
            return reply

        raise HTTPException(status_code=404, detail=f"reply {id} not found")

    except:
       raise HTTPException(status_code=404, detail=f"tweet {id} not found")
        #print("I am lost")
@router.get(
    "/{id}", response_description="Get a single tweet"
)
async def get_tweet(id: str,user: str = Depends(get_current_user)):
    if(len(id)!=24):
        raise HTTPException(status_code=404, detail=f" tweet id must have 24 length")
    id=ObjectId(id)
    try:
    
        tweet=main.Tweet.find_one({"_id":  id})
        if tweet is not None:
            replies= list(main.Replies.find())

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
        
            tweet['_id']=str( tweet['_id'])
            tweet['user']=str(tweet['user'])
            tweet['created_at']=str(tweet['created_at'])
            tweet['updated_at']=str(tweet['updated_at'])
            tweet.setdefault('replies',[])
            replies=Rearrange_again(replies)
            for rep in replies:
                if rep['tweet_id']==tweet['_id']:
                    tweet['replies'].append(rep)
            #tweet=json.dumps(tweet)
            return tweet


        raise HTTPException(status_code=404, detail=f"tweet {id} not found")

    except:
    
     return get_reply(id,user)

    
@router.put("/{id}", response_description="Update a tweet", response_model=schemas.TweetBaseSchema)
async def update_tweet(id: str, tweet: schemas.UpdateTweetSchema = Body(...),user: str = Depends(get_current_user)):
    tweet = {k: v for k, v in tweet.dict().items() if v is not None}
    if(len(id)!=24):
        raise HTTPException(status_code=404, detail=f" tweet id must have 24 length")
    
    if len(tweet) >= 1:
        update_result =  main.Tweet.update_one({"_id": ObjectId(id)}, {"$set": tweet})

        if update_result.modified_count == 1:
            if (
                updated_tweet := main.Tweet.find_one({"_id": id})
            ) is not None:
                return updated_tweet

    if (existing_tweet := main.Tweet.find_one({"_id": ObjectId(id)})) is not None:
        return existing_tweet

    raise HTTPException(status_code=404, detail=f"tweet {id} not found")

