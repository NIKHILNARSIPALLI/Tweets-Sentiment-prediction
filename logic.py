#This file contains the logic behind the predictions of the tweets

import numpy as np
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.linear_model import LogisticRegression

#ignore warnings
import warnings
warnings.simplefilter(action='ignore')


def stemming(content):
  snow_stemmer = SnowballStemmer(language='english')
  stemmed_content = re.sub(r'(@)[\s]','',content) #Here we are removing the name tags in the tweet that does not contribute to sentiment
  stemmed_content = re.sub(r'http\S+',' ',stemmed_content) #remonving any website links
  stemmed_content = re.sub(r'www\.\S+',' ',stemmed_content) #removing anything that starts with wwww
  stemmed_content = re.sub(r'[^A-Za-z]',' ',stemmed_content) #removing any characters that are not alphabets
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [snow_stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)

  return stemmed_content

def predicts(text : str):
  model = pickle.load(open('data/model/Tweets_Sentiment_Analysis_Trained_Model_Logistic_Regression.sav','rb'))
  vectorizer = pickle.load(open('data/vectorizer/Tweets_Sentiment_Analysis_count_vectorizer.pkl','rb'))

  input_text = stemming(text)
  input_text = vectorizer.transform([input_text])
  predict = model.predict(input_text)
  
  if predict == 0:
    return 'Negative'
  else:
    return 'Positive'
  

if __name__ == '__main__':
  while True:
    user_input = str(input("Enter"))
    if user_input.lower() == 'quit':
      break
    print(predicts(user_input))