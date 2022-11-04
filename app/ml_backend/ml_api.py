# text preprocessing modules
from string import punctuation
# text preprocessing modules
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
from os.path import dirname, join, realpath
import os
import joblib
from fastapi import APIRouter
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fastapi import FastAPI
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

from gensim.matutils import softcossim 

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

@router.post("/create_recommendation")
def create_recommendation(sentence: str,corpus:list,limit:int):
    """
    A simple function that receive a sentence content and predict the other related texts
    :param sentense:
    :return: sentense, list of other simillar sentences,limit of similaraties
    """
    # clean the review
    #cleaned_sentence = text_cleaning(sentence)
    # for sent in corpus:
    #     text_cleaning(sent)
    if sentence not in corpus:
        corpus.append(sentence) 
    cosine_sim=cosineSimillarity(corpus)
    print(cosine_sim)
    print("Sentence: ",sentence)
    recommendation=get_recommendations(sentence,cosine_sim, corpus,limit)
    print(recommendation)
    recommendationList=[]
    print(type(recommendation))
    # for key,value in recommendation:
    #     print(value)
    #     recommendationList.append(value)
    recommendationList=list(recommendation.values())
    print(recommendationList)
    print("*******************************************************************************************\n\n")
     
    return recommendationList

def loadReframerModel():
    # load the model from disk

    filename = 'output/t5_controlled_model.pkl'
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









