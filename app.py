# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 19:13:46 2020

@author: LRG
"""

import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
import re
#import nltk

#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer

app=Flask(__name__)
model=pickle.load(open('restaurant_model.pkl','rb'))

cv=pickle.load(open('transform.pkl','rb'))

@app.route('/',methods=['GET'])
def home():
    return "<h1>NLP-APP</h1>"

@app.route('/predictor',methods=['POST'])
def predictor():
    st_array=list(request.get_json(force=True).values())[0]
    corpus=[]
    for st in st_array:
        review=re.sub('[^a-zA-z]',' ',st)
        review=review.lower()
        review=review.split()
        #ps=PorterStemmer()
        review = [word for word in review if not word in ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]]
        review=' '.join(review)
        corpus.append(review)
    trail_x=cv.transform(corpus).toarray()
    trail_pred=model.predict(trail_x)
    lst=[]
    for i in trail_pred:
        lst.append(int(i))
    return jsonify(outputs=lst)
    #return(trail_pred)


#predictor(["so nice","food is bad"])
    
if __name__=="__main__":
    app.run(debug=False)
