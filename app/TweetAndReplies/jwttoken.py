from datetime import datetime, timedelta
import email
from jose import JWTError, jwt
import TweetAndReplies.main
import json


SECRET_KEY = "b0635b0b4d0ffd56c85059e9b6051b75c9db1c3e9511f29b7466178ca1e8eb92"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token:str,credentials_exception):
     token_data=None
     try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        data:main.UserModel = payload.get("sub")
        data=data.replace("\'","\"")
        print(data)
        print(type(data))
        data=json.loads(data)
        print(type(data))
        if data is None:
            raise credentials_exception
        token_data = main.TokenData(email=data['email'],username=data['username'],password=data['password'],id=data['id'])
        print(token_data.username)
     except JWTError:
        raise credentials_exception
     return token_data


    
    