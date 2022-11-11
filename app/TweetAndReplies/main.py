from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Request,status,Body,APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from TweetAndReplies.hashing import Hash
from TweetAndReplies.jwttoken import create_access_token
from TweetAndReplies.oauth import get_current_user
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import TweetAndReplies.schemas
from fastapi_jwt_auth import AuthJWT
from bson import ObjectId
from fastapi import Depends,HTTPException,status
from TweetAndReplies.jwttoken import verify_token
from fastapi.security import OAuth2PasswordBearer
import json

router = APIRouter()

from pymongo import MongoClient
mongodb_uri = 'mongodb://Kclaude:Kwizeraeager@localhost:27017/Reframing?authSource=admin'
remote_mongo_db_uri='mongodb+srv://kclaude:Kwizeraeager@positive-reframing-twee.bjvzmtp.mongodb.net/Reframing'
port = 8000
client = MongoClient(remote_mongo_db_uri, port)
db = client["Positive_reframing"]
User=db['users']
Tweet=db['tweets']
Replies=db['replies']


class UserModel(BaseModel):
    email:str
    username: str
    password: str
    profile_image:str
    
class Login(BaseModel):
	username: str
	password: str
class Token(BaseModel):
    access_token: str
    token_type: str
class TokenData(UserModel):
    id:Optional[str]=None

def userEntity(user) -> dict:
    return {
        "id": str(user["_id"]),
        "name": user["username"],
        "email": user["email"],
        "password": user["password"],
        "profile_image": user["profile_image"],
        "created_at": user["created_at"],
        "updated_at": user["updated_at"]
    }


def userResponseEntity(user) -> dict:
    return {
        "id": str(user["_id"]),
        "name": user["username"],
        "email": user["email"],
        "profile_image": user["profile_image"],
        "created_at": user["created_at"],
        "updated_at": user["updated_at"]
    }


def embeddedUserResponse(user) -> dict:
    return {
        "id": str(user["_id"]),
        "name": user["username"],
        "email": user["email"],
         "profile_image": user["profile_image"]
    }


def userListEntity(users) -> list:
    return [userEntity(user) for user in users]

def tweetEntity(tweet) -> dict:
    return {
        "id": str(tweet["_id"]),
        "category": tweet["category"],
        "message": tweet["message"],
        "attachements": tweet["attachements"],
        "hashtags":tweet["hashtags"],
        "timeLeft":tweet["timeLeft"],
        "isreacted":tweet["isreacted"],
        "count":tweet["count"],
        "mentions":tweet["mentions"],
        "displayReplies":tweet["displayReplies"],
        "retweets":tweet["retweets"],
        "user": str(tweet["user"]),
        "created_at": tweet["created_at"],
        "updated_at": tweet["updated_at"]
    }


def populatedTweetEntity(tweet) -> dict:
    return {
        "title": tweet["title"],
        "category": tweet["category"],
        "message": tweet["message"],
        "attachements": tweet["attachements"],
        "hashtags":tweet["hashtags"],
        "timeLeft":tweet["timeLeft"],
        "isreacted":tweet["isreacted"],
        "count":tweet["count"],
        "mentions":tweet["mentions"],
        "displayReplies":tweet["displayReplies"],
        "retweets":tweet["retweets"],
        "created_at": tweet["created_at"],
        "updated_at": tweet["updated_at"],
        "user": embeddedUserResponse(tweet["user"]),
        
    }
class UserNotFound(Exception):
    pass


def tweetListEntity(tweets) -> list:
    return [populatedTweetEntity(tweet) for tweet in tweets]


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_current_user(token: str = Depends(oauth2_scheme)):
	credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

	return verify_token(token,credentials_exception)
def require_user(Authorize: AuthJWT = Depends()):
    try:
        Authorize.jwt_required()
        user_id = Authorize.get_jwt_subject()
        user = userEntity(User.find_one({'_id': ObjectId(str(user_id))}))

        if not user:
            raise UserNotFound('User no longer exist')


    except Exception as e:
        error = e.__class__.__name__
        print(error)
        if error == 'MissingTokenError':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail='You are not logged in')
        if error == 'UserNotFound':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail='User no longer exist')
        if error == 'NotVerified':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail='Please verify your account')
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail='Token is invalid or has expired')
    return user_id
@router.get("/")
def read_root(current_user:UserModel = Depends(get_current_user)):
	return {"data":current_user}