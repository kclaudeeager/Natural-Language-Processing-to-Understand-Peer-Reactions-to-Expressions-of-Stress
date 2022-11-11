from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Request,status,Body,APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from TweetAndReplies.hashing import Hash
from TweetAndReplies.jwttoken import create_access_token
from TweetAndReplies.oauth import get_current_user
from datetime import datetime, timedelta
import TweetAndReplies.schemas as schemas
import TweetAndReplies.main as main
from bson import ObjectId
router = APIRouter()
@router.post('/register')
def create_user(request:main.UserModel):
    user_object = dict(request)
    user_object.setdefault('created_at','')
    user_object.setdefault('updated_at','')
    hashed_pass=Hash.bcrypt(request.password)
    user_object["password"] = hashed_pass
    user = main.User.find_one({"username":user_object["username"]})
    if user:
         raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail='Account already exist')

    
    user_object['created_at']=(datetime.utcnow())
    user_object['updated_at']=user_object['created_at']
    user_id = main.User.insert_one(user_object)
    return {str(user_object):"created"}
	

@router.post('/login')
def login(request:main.OAuth2PasswordRequestForm = Depends()):
	user = main.User.find_one({"username":request.username})

	if not user:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail = f'No user found with this {request.username} username')
	if not Hash.verify(user["password"],request.password):
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail = f'invalid credentials ')
    
	access_token = create_access_token(data={"sub": str({"username":user["username"],"id":str(user["_id"]),"email":user["email"],"password":user["password"],"profile_image":user["profile_image"]})})
    
	return {"access_token": access_token, "token_type": "bearer"}
@router.get('/',response_description="Get all user")
def findAll(userReturned: str = Depends(get_current_user)):
    if(userReturned.username!="Claude"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail = f'Unauthorized to use this endpoint')
    users=list(main.User.find())
    for user in users:
        user['_id']=str(user['_id'])
        
    print("users  ",users)
    return {'status': 'success', 'users': users}
@router.get("/{id}",response_description="Get a single user")
def findUser(id: str,userReturned: str = Depends(get_current_user)):
    if(len(id)!=24):
            raise HTTPException(status_code=404, detail=f" user id must have 24 length")
    
    try:
        id=ObjectId(id)
        if (user :=  main.User.find_one({"_id":  id})) is not None:
            user['_id']=str(user['_id'])
            return {'status':'success','user':user}

        raise HTTPException(status_code=404, detail=f"user {id} not found")

    except:
      raise HTTPException(status_code=404, detail=f"user {id} not found")
   
