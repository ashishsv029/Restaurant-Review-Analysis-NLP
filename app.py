# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 19:13:46 2020

@author: LRG
"""

import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app=Flask(__name__)
model=pickle.load(open('restaurant_model.pkl','rb'))

cv=pickle.load(open('transform.pkl','rb'))

@app.route('/predictor',methods=['POST'])
def predictor():
    st_array=list(request.get_json(force=True).values())[0]
    corpus=[]
    for st in st_array:
        review=re.sub('[^a-zA-z]',' ',st)
        review=review.lower()
        review=review.split()
        ps=PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
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
