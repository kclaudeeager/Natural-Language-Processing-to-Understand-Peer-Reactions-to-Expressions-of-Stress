# text preprocessing modules
from string import punctuation
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import os
from os.path import dirname, join, realpath
import joblib
import uvicorn
from fastapi import FastAPI 
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI(
    title="recommendation API",
    description="A simple API that use NLP model to generate similarity and create a recommendation",
    version="0.1",
)

# # load the sentiment model
# with open(
#     join(dirname(realpath(__file__)), "models/trained_model1.pkl"), "rb"
# ) as f:
#     model = joblib.load(f)

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
def get_key(val,my_dict):
    #print(my_dict.items())
    
    for key, value in my_dict.items():
         #print(key,": "+value)
         print(val)
         if val.lower() in value.lower():
             print("OKKKKKKK"+val)
             print(key)
             return key
         else:
             print(val.__eq__(value))
             print("Not equal in any way")
 
    return "key doesn't exist"
def getTweetDict(tweetList):
    myDict=dict()
    for i in range(0,len(tweetList)):
        myDict.update({i:tweetList[i]})
        #print(myDict)
    return myDict
def get_recommendations(title, cosine_sim, tweetList,limit):
    # Get the index of the tweet that matches the title
    twitDict=getTweetDict(tweetList)
    print(twitDict)
    # print(title)
    # print(tweetList[0])
    # print(twitDict.get(0))
    idx=get_key(title,twitDict)
    #print(tweetList[124])
    # print(title)
    # print(twitDict.get(idx))
    #print(idx)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores,key=lambda x:x[1],reverse=True)

    #print(len(sim_scores))
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

    # Generate the tf-idf vectors for the corpus
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # compute and print the cosine similarity matrix
    #cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    #print(cosine_sim)
    # Print time taken
    print("Time taken: %s seconds" % (time.time() - start))
    return cosine_sim
@app.post("/create_recommendation")
def create_recommendation(sentence: str,corpus:list):
    """
    A simple function that receive a sentence content and predict the other related texts
    :param sentense:
    :return: sentense, list of other simillar sentences
    """
    # clean the review
    #cleaned_sentence = text_cleaning(sentence)
    # for sent in corpus:
    #     text_cleaning(sent)
    cosine_sim=cosineSimillarity(corpus)
    print(cosine_sim)
    print("Sentence: ",sentence)
    recommendation=get_recommendations(sentence,cosine_sim, corpus,5)
    print(recommendation)
    print("*******************************************************************************************\n\n")
  
    return recommendation