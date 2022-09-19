from sys import prefix
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import TweetAndReplies.main
from TweetAndReplies.routes import tweet,user,reply
from ml_backend import ml_api
app = FastAPI(
    title="Sentiment Model API",
    description="An API that use NLP model to predict the sentiment of the the tweet replies , and also generate  similarity and create a recommendation \n after all reframe positively the negative sentiments",
    version="0.1",
)
origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:8085"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(user.router, tags=['Users'], prefix='/api/users')
app.include_router(tweet.router, tags=['Tweets'], prefix='/api/tweets')
app.include_router(reply.router,tags=['Replies'], prefix='/api/replies')
app.include_router(ml_api.router,tags=['Reframing'], prefix='/api')

@app.get("/")
def runRoot():
    return {"message":"this is positive reframing api"}


@app.get("/api/positive-reframing")
def root():
    return {"message": "Welcome Positive reframing"}

