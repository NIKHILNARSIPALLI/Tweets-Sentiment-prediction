# Tweets-Sentiment-prediction
This program is  made using a machine learning model to understand the emotion behind a text tweet.
Above program is a Machine Learning program that predicts the sentiment of the data based on the learning done by the model using Sentimet 140 dataset

## Check the deployed application:
https://tweets-sentiment-prediction-by-nikhil-narsipalli.streamlit.app/

## Importing the Dataset
The First thing to do is to link the kaggle API to our working environment.

For this let us install kaggle library into out environment

```python
# install Kaggle library
!pip install kaggle
```

Now let's link the kaggle.json file that availale in your kaggle account. This json file allows us to access Kaggle API to directly download the large datasets into cloud IDE like colbas, which would rather take extensive amount of time.

Here, we are creating a new path for the json file and changing its persmissions for access

```python
#configuring the path for kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

Now let's import the Sentiment 140 dataset which contains 16 million tweets that has 2 target values in them. One is positive - value 4, another is negative - value 0

i.e, Importing the dataset

```python
#API to fetch the dataset from kaggle
!kaggle datasets download -d kazanova/sentiment140
```

The dowloaded dataset is a compressed dataset, since it was downloaded using a API. Now, we are going to expand the file to work with the dataset

```python
#Extracting the compressed dataset
from zipfile import ZipFile
dataset = '/content/sentiment140.zip'

with ZipFile(dataset,'r') as zip:
zip.extractall()
print("The File has been extracted successfully")
```

Now that the dataset has been successfully imported and extracted, we can proceed to importing required dependencies for the project

```python
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score)
```

After importing the dependencies, we can continue towards data pre processing. This step is particularly important as it has a major impact on the accuracy of the model.

## Data Pre-processing
First let us start by importing the dataset as pandas dataframe and giving it a list of column names. This helps us in clearly distinguishing the data and helps in manipulating it easily.

```python
# Importing data
column_names = ['target','id','date','flag','user','text']
tweets_data = pd.read_csv("/content/training.1600000.processed.noemoticon.csv",names=column_names,encoding='ISO-8859-1')
```

Now, let us understand the dataframe we have imported to process it. Let's do this by observing the shape of the dataframe and checking any null values present.

```python
# check data

tweets_data.head()
tweets_data.shape

# Cheking for null values in the columns
tweets_data.isnull().sum()

# Cheking the distribution of the target column
tweets_data['target'].value_counts()
```

From the above code we can concur that the dataset has no null values, so no need to adjust the data for null values. Since, this is a text data with only 2 classifications (0 = Negative, 4 = Positive) we do not have any outliers. #

While doing the above step let us also convert the value 4 = Positive to 1 = Positive. This helps us in directly feeding into the machine learning model and gets in norms with usual industry standards.

```python
#Coverting target value 4 to 1
tweets_data.replace({'target': {4:1}},inplace=True)

#let's check the coversion
tweets_data['target'].value_counts()
```

Now that the target value has been converted, we can proceed with defining a function called stemming, that removes any unwanted data in a text like url links, @names that do not contain any useful emotion data in most of the cases. Doing this helps us in reducing the complexity of the model and makes the data a lot simpler.

The function also removes any stopwords present in the data, along with converting the lexicons into their root words. We are also downloading stopwords data at the start and initialzing the lemmatizer that is responsible for the conversion of root words.

```python
# Downloading stopwords
nltk.download('stopwords')

# Initialzing the lemmatizer
snow_stemmer = SnowballStemmer(language='english')

# Defining the stemming function
def stemming(content):
    stemmed_content = re.sub(r'(@)[\s]','',content) #Here we are removing the name tags in the tweet that does not contribute to sentiment
    stemmed_content = re.sub(r'http\S+',' ',stemmed_content) #remonving any website links
    stemmed_content = re.sub(r'www\.\S+',' ',stemmed_content) #removing anything that starts with wwww
    stemmed_content = re.sub(r'[^A-Za-z]',' ',stemmed_content) #removing any characters that are not alphabets
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [snow_stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content
```

Coming to the final step in data pre-processing we will apply the above function to all the text values in the target column of the dataset and create a new column to process the data into the machine learnign model.

```python
# Creating new column with stemmed data

tweets_data['stemmed_data'] = tweets_data['text'].apply(stemming)
```

## Making Train Test Split

Now, that the pre-processing of the data is done. We can now proceed to splitting the data into training data and testing data. These training and testing data are then converted in arrays for the machine learning model to undestand them.

Before doing the above, let us first confirm the distribution of the target values in the dataset. Having a even distribution contributes to better accuracy. So, let's make sure that the data is evenly distributed

```python
# Cheking the distribution of the target column

tweets_data['target'].value_counts()
```

My data shows a even distribution among the classes, so no need to further distribute the data #

```python
# Splitting the data into train test split

X = tweets_data['stemmed_data'].values
y = tweets_data['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 2)
```

In the above line of code we have taken the stemmed_data column from the dataset that is newly created and then proceeded to convert 80% of the values to training data and 20% of the values for testing data with a random_state =2 and stratify =y. Stratify = y makes sure that the splitting of the data is evenly distrbiuted along the y values,in this case targets mentiong a random state in the code without leaving it to default will help the algorithm fetch the same type of set of values every time it runs and just changes the content of those values.

Now let's convert the textual data into a list of arrays. For this we use vectorizers, first we will fit the vectorizer with the training data and then use those numerical values to transform the test data. Vectorizer gives each word a token/numberical values and maps it like a indexed dictionary, Now this mapped dictionary will be used to convert the test data to arrays

```python
#converting the text data into numerical data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
```

## Training the model

Now that we have a list of arrays that can be feeded into a machine learning model to predict the emotion of the tweets. Let us start by creating a Logistic Regression model and feeding the training data into it.

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

Now let us check the training accuracy of the model:

```python
# Training Accuracy
X_train_predictions = model.predict(X_train)
training_data_accuracy = accuracy_score(y_train, X_train_predictions)

print("The training accuracy of the model is: ",training_data_accuracy)
```

Let us also check the accuracy of the test data:

```python
#Accuracy score of test data
X_test_predictions = model.predict(X_test)
test_data_accuracy = accuracy_score(y_test,X_test_predictions)

print("The Test accuracy is : ",test_data_accuracy)
```

Since, the testing acccuray is along 78% and the training accuracy is along 82% the model is considered not overfited and a good model.

## Predictions

Now let us create a while loop that takes input from the users then pre-process the data and give us the predictions of the input text

For this let us create a function that does this whole process and then later we can simply call the function to predict the data:

```python
def predict(input_content : str):
    input_content = stemming(input_content)
    input_content = loaded_vectorizer.transform([input_content])
    predicts = (loaded_model.predict(input_content))

    if predicts == 0:
        return 'Negative'
    else:
        return 'Positive'
```

Let us create a while loop to take continous inputs from user to predict the emotion and also give the user an option to stop when types in the word 'quit'

```python
while True:
    user_input = str(input())
    if user_input.lower() == 'quit':
        break
    print(predict(user_input))
```

## Saving the model

Now that the model has been successfully trained let us save it future uses. The saving process includes saving the trained model and the vectorizer dictionary to run the converted text into the saved model. For this we use pickle to save them and extract them for future uses.

```python
import pickle

# Saving the model            
filename = 'Tweets_Sentiment_Analysis_Trained_Model_Logistic_Regression.sav'
pickle.dump(model,open(filename,'wb'))

#saving the vectorizer
with open('Tweets_Sentiment_Analysis_count_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
```

## Conclusion

The basic idea of this program is to let the machine understand and predict the emotion of a tweet. This model is trained using Logistic Regression with a training accuracy of 82.82% and a testing accuracy of 77.52%. This code is credited by @NikhilNarsipalli - github
