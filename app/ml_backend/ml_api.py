# text preprocessing modules
from http.client import HTTPException
import json
from string import punctuation
from TweetAndReplies.schemas import TweetBaseSchema
from TweetAndReplies.oauth import get_current_user
from TweetAndReplies.routes.tweet import getTweets
# text preprocessing modules
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
from os.path import dirname, join, realpath
import os
import joblib
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fastapi import  HTTPException, Depends,APIRouter
import pickle
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import gensim
# upgrade gensim if you can't import softcossim
from gensim.models import TfidfModel
from gensim.matutils import softcossim 
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess
import numpy as np
print(gensim.__version__)
from typing import List
from gensim.matutils import softcossim 
from bson import ObjectId
import TweetAndReplies.main as main
from TweetAndReplies.routes.reply import Rearrange_again

# # Download the FastText model
# fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')

# with open(join(dirname(realpath(__file__)), "models/fasttext_model300.pkl"), 'wb') as f:
#     pickle.dump(fasttext_model300, f)

# if os.path.getsize(join(dirname(realpath(__file__)), "models/fasttext_model300.pkl")) > 0:      
#     with open(join(dirname(realpath(__file__)), "models/fasttext_model300.pkl"), "rb") as f:
#         unpickler = pickle.Unpickler(f)
#         # if file is not empty scores will be equal
#         # to the value unpickled
#         fasttext_model300 = unpickler.load()
# print("File has successfully saved")
tweet_list=[]
replies_list=[]
router = APIRouter()

@router.get('/')
def get_root():
    return {'message': 'Welcome to the positive reframing api'}
# Download the FastText model
#fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')

# load the sentiment model

with open(
    join(dirname(realpath(__file__)), "models/trained_model1.pkl"), "rb"
) as f:
    model = joblib.load(f)


# cleaning the data
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
    
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        # load stopwords
        stop_words = stopwords.words("english")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
        
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
        
    # Return a list of words
    return text

@router.get("/predict-replies")
def predict_sentiment(reply: str):


    """
    A simple function that receive a reply content and predict the sentiment of the content.
    :param reply:
    :return: prediction, probabilities
    """
    # clean the review
    cleaned_review = text_cleaning(reply)
    
    # perform prediction
    prediction = model.predict([cleaned_review])
    print("Prediction : ",prediction)
    output = int(prediction[0])
    probas = model.predict_proba([cleaned_review])
    output_probability = "{:.2f}".format(float(probas[:, output]))
    
    # output dictionary
    sentiments = {-1.0: "Negative", 1.0: "Positive",0.0:"Neutral"}
    
    # show results
    result = {"prediction": sentiments[output], "Probability": output_probability}
    return result



def get_key(val,my_dict):
    #print(my_dict.items())
    
    for key, value in my_dict.items():
         #print(key,": "+value)
         print(val)
         if val.lower() in value.lower():
            #  print("OKKKKKKK"+val)
            #  print(key)
             return key
        #  else:
        #      print(val.__eq__(value))
        #      print("Not equal in any way")
 
    return "key doesn't exist"
def getTweetDict(tweetList):
    # Prepare a dictionary and a corpus.
    # dictionary = corpora.Dictionary([simple_preprocess(doc) for doc in tweetList])
    # print("created dict ",dictionary)
    # Prepare the similarity matrix
    # similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100)
    # print(similarity_matrix)
    myDict=dict()
    for i in range(0,len(tweetList)):
        myDict.update({i:tweetList[i]})
        #print(myDict)
    return myDict
def get_recommendations(title, cosine_sim, tweetList,limit=2):
    # Get the index of the tweet that matches the title
 
    
    twitDict=getTweetDict(tweetList)
    print(twitDict)
 
    idx=get_key(title,twitDict)
   
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores,key=lambda x:x[1],reverse=True)

    #print(len(sim_scores))
    if limit>=len(tweetList):
        print("Limit ecceded!!")
        limit=2
    sim_scores = sim_scores[1:limit]
    #print(sim_scores)
    # Get the tweet indices
    tweet_indices = [i[0] for i in sim_scores]
    #print(tweet_indices)
    # Return the top 10 most similar tweets
    newDict={index:twitDict[index] for index in tweet_indices}
    return newDict
def cosineSimillarity(corpus):
    
    start = time.time()
    # Initialize an instance of tf-idf Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    dictionary = corpora.Dictionary([simple_preprocess(doc) for doc in corpus])
    # Prepare the similarity matrix
    # similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100)
    
    # len_array = np.arange(len(corpus))
    # xx, yy = np.meshgrid(len_array, len_array)
    # cossim_mat = pd.DataFrame([[round(softcossim(corpus[i],corpus[j], similarity_matrix) ,2) for i, j in zip(x,y)] for y, x in zip(xx, yy)])
   
    # Generate the tf-idf vectors for the corpus
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    #tfidf_matrix=TfidfModel(corpus)
    doc_term_matrix = tfidf_matrix.todense()
    df = pd.DataFrame(doc_term_matrix, 
                  columns=tfidf_vectorizer.get_feature_names(), 
                  index=[corpus])
    display(df)
    # compute and print the cosine similarity matrix
    #cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    #cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    cosine_sim = linear_kernel(df, df)
    print(cosine_sim)
    # Print time taken
    print("Time taken: %s seconds" % (time.time() - start))
    return cosine_sim

#Returning the list of tweets with thier replies after similarity computation
def get_grouped_tweet_with_replies_after_similarity_calculation(tweetDict,similarTweets_dict):
#     returnedGroupList=[]
    similarTweets_dict_copy=similarTweets_dict
    for tweet in tweetDict:
        for key,value in similarTweets_dict.items():
#             print("value: ",value)
#             print("tweet text: ",tweet.get('text'))
            if tweet['message']==value:
                #print("found...: ",tweet)
                value=tweet
                similarTweets_dict[key]=value
#                 print("Value now: >>>",value)
#                 print("similarTweets_dict now: <<<<>>>>>>",similarTweets_dict)
    return similarTweets_dict_copy,similarTweets_dict
#adding the replies of the similar tweets  to the tweet in that group
def addRepliesToTweet(similar_group_tweets):
    tweetGroup=similar_group_tweets['tweet_group']
    orginal_tweet=similar_group_tweets['tweet']
    replies=orginal_tweet['replies']
    for replyObj in tweetGroup:
        newdict=tweetGroup[replyObj]
        #print("New dict replies:::",[reply for reply  in newdict['replies']])
        #Retturning only the postive replies for each tweet
        bestReplies=[reply for reply  in newdict['replies'] if reply["classification"].get('prediction')=='Positive']
        print("Best replies::",bestReplies)
        for reply in bestReplies:
            if reply not in replies:
             
                replies.append(reply)
    orginal_tweet['replies']=replies
    similar_group_tweets['tweet']=orginal_tweet

   

@router.get("/create_recommendation",response_description=" A simple function that receive a sentence content and predict the other related texts :param sentense::return: sentense, list of other simillar sentences,limit of similaraties")
def create_recommendation(id: str,limit:int):
    """
    A simple function that receive a sentence content and predict the other related texts
    :param sentense:
    :return: sentense, list of other simillar sentences,limit of similaraties
    """
    # clean the review
    #cleaned_sentence = text_cleaning(sentence)
    # for sent in corpus:
    #     text_cleaning(sent)
    if(len(id)!=24):
            raise HTTPException(status_code=404, detail=f" tweet id must have 24 length")
    id=ObjectId(id)
    
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
    tweetList=getTweets() 
    #print("Tweet>>",tweet)  
    sentence=tweet['message']
   # print("Sentence::",sentence)
    corpus=[eachTweet['message'] for eachTweet in tweetList if eachTweet['message']!=sentence ]
    if sentence not in corpus:
        corpus.append(sentence) 
    cosine_sim=cosineSimillarity(corpus)
    print(cosine_sim)
   # print("Sentence: ",sentence)
    recommendation=get_recommendations(sentence,cosine_sim, corpus,limit)
    #print(recommendation)
    similarTweets_dict=recommendation
    similarTweets_dict_copy,similarTweets_dict=get_grouped_tweet_with_replies_after_similarity_calculation(tweetList,similarTweets_dict)
    print(similarTweets_dict_copy,'\n___________________________________\n')
    #print("reformed: ",similarTweets_dict)
    tweet_with_group_dict={"tweet":tweet,"tweet_group":similarTweets_dict}
    print("tweet_with_group_dict: \n",tweet_with_group_dict)
    print("\n___________________________________________\n")
    addRepliesToTweet(tweet_with_group_dict)
    returnedTweet=tweet_with_group_dict.get('tweet')
    print(returnedTweet)
    print("*******************************************************************************************\n\n")
     
    return returnedTweet

def loadReframerModel():
    # load the model from disk

    filename = 'output1/t5_controlled_model.pkl'
    loaded_reframer = pickle.load(open(filename, 'rb'))
    return loaded_reframer 
loaded_reframer=loadReframerModel()
def reframeTexts(text_list,loaded_reframer):
    reframed_texts = [loaded_reframer(phrase)[0]['summary_text'] for phrase in text_list]
    print(reframed_texts)
    return reframed_texts
@router.post("/reframe")
def reframe_text(texts: list):
    """
    A simple function that receive a text content and create positive reframing corresponding to that text.
    :param text:
    :return: reframed_text, reframing_strategy
    """
    reframed_texts=reframeTexts(texts,loaded_reframer)
    result={"reframed_texts":reframed_texts}
    return result









