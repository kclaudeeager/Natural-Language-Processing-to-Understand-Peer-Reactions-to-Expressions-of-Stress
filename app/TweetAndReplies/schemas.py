from ctypes import py_object
from datetime import datetime
from numbers import Number
from pyexpat import model
from typing import List, Union
from unicodedata import numeric
from pydantic import BaseModel, EmailStr, constr,Field
from bson import ObjectId
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class UserBaseSchema(BaseModel):
    name: str
    email: str
    profile_image: str
    role: str = None
    created_at: datetime = None
    updated_at: datetime = None

    class Config:
        orm_mode = True


class CreateUserSchema(UserBaseSchema):
    password: constr(min_length=8)
    passwordConfirm: str
 
class UserInDbSchema(UserBaseSchema):
    password: constr(min_length=8)

class LoginUserSchema(BaseModel):
    email: EmailStr
    password: constr(min_length=8)


class UserResponseSchema(UserBaseSchema):
    id: str
    pass


class UserResponse(BaseModel):
    status: str
    user: UserResponseSchema


class FilteredUserResponse(UserBaseSchema):
    id: str

class attachement(BaseModel):
    caption:str
    name:str
    src:str
class TweetBaseSchema(BaseModel):
    mentions:list
    message:str
    hashtags:list
    attachements:List
    timeLeft:str
    isreacted:bool=False
    count:int=0
    displayReplies:bool=False
    retweets:int=0
    created_at: datetime = None
    updated_at: datetime = None
    replies:List=[]
    
    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class CreateTweetSchema(TweetBaseSchema):
    user:PyObjectId=None
    pass


class TweetResponse(TweetBaseSchema):
    id: str
    user: FilteredUserResponse
    created_at: datetime
    updated_at: datetime


class UpdateTweetSchema(BaseModel):
    mentions:list=None
    message:str=None
    hashtags:list=None
    timeLeft:str=None
    isreacted:bool=False
    count:int=0
    displayReplies:bool=False
    retweets:int=0
    attachements:List
    user: str = None
    replies:Union[List,None]=None
    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ListPostResponse(BaseModel):
    status: str
    results: int
    posts: List[TweetResponse]
class RepliesBaseSchema(BaseModel):
    tweet_id:str
    mentions:list=None
    message:str=None
    hashtags:list=None
    timeLeft:str=None
    isreacted:bool=False
    count:int=0
    displayReplies:bool=False
    retweets:int=0
    attachements:List
    user: str = None
    replies:Union[List,None]=None
    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
class CreateReplySchema(RepliesBaseSchema):
    user: ObjectId = None
    tweet:ObjectId =None