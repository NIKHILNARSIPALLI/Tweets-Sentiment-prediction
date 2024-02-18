import streamlit as st
import logic
    
#app logic for prediction
def launch(content):
    result = logic.predicts(content)

    if result == 'Positive':
        st.text(content)
        st.write('The Above tweets is Positive :thumbsup:')
    else:
        st.text(content)
        st.write('The Above tweet is Negative :thumbsdown:')

st.set_page_config(layout='centered')

st.title("Find out your 'Tweets' emotion :hushed:")
st.markdown('#')
st.header(":arrow_down:")
st.markdown('''Enter a Tweet below to understand the emotion behind the tweet by [@NikhilNarsipalli - github](https://github.com/NIKHILNARSIPALLI)''')
input_text = st.text_area('',placeholder='Enter here',height=400,max_chars=200)

st.markdown('##')

if st.button('Predict',key = 'predict_button',use_container_width=True):
    if input_text.strip():
        launch(input_text)
    else:
        st.write(":crossed_swords: Enter text to predict :crossed_swords:")

if st.button('Check Code',key='check_code',use_container_width=True):

    st.header("Code Explanation")
    
    st.write("Above program is a Machine Learning program that predicts the sentiment of the data based on the learning done by the model using Sentimet 140 dataset")
    st.markdown('#')

    st.subheader('Importing the Dataset')

    st.write('The First thing to do is to link the kaggle API to our working environment.')
    st.write('For this let us install kaggle library into out environment')
    st.code("""
                # install Kaggle library
                !pip install kaggle
                """,line_numbers=True)
    
    st.markdown("""
                #
                Now let's link the kaggle.json file that availale in your kaggle account.
                This json file allows us to access Kaggle API to directly download the large
                datasets into cloud IDE like colbas, which would rather take extensive amount of time.
            
                Here, we are creating a new path for the json file and changing its persmissions for access""")
    st.code('''
            #configuring the path for kaggle.json
            !mkdir -p ~/.kaggle
            !cp kaggle.json ~/.kaggle/
            !chmod 600 ~/.kaggle/kaggle.json''',line_numbers=True)
    

    st.markdown('''
                #
                Now let's import the Sentiment 140 dataset which contains 16 million tweets that has 2 target values in them.
                One is positive - value 4, another is negative - value 0
                
                i.e, Importing the dataset''')
    st.code('''
            #API to fetch the dataset from kaggle
            !kaggle datasets download -d kazanova/sentiment140''',line_numbers=True)
    
    st.markdown('''
                #
                The dowloaded dataset is a compressed dataset, since it was downloaded using a API.
                Now, we are going to expand the file to work with the dataset''')
    st.code('''
            #Extracting the compressed dataset
            from zipfile import ZipFile
            dataset = '/content/sentiment140.zip'

            with ZipFile(dataset,'r') as zip:
            zip.extractall()
            print("The File has been extracted successfully")
            ''',line_numbers=True)
    

    st.markdown('''
                #
                Now that the dataset has been successfully imported and extracted, we can proceed to 
                importing required dependencies for the project''')
    st.code('''
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
            ''',line_numbers = True)
    

    st.markdown('''
                #
                After importing the dependencies, we can continue towards data pre processing.
                This step is particularly important as it has a major impact on the accuracy of the model.
                #''')
    
    st.subheader('Data Pre-processing')

    st.markdown('''First let us start by importing the dataset as pandas dataframe and giving it a list of column names.
                This helps us in clearly distinguishing the data and helps in manipulating it easily.''')
    st.code('''
            # Importing data
            column_names = ['target','id','date','flag','user','text']
            tweets_data = pd.read_csv("/content/training.1600000.processed.noemoticon.csv",names=column_names,encoding='ISO-8859-1')
            ''',line_numbers=True)
    

    st.markdown('''
                #
                Now, let us understand the dataframe we have imported to process it. Let's do this by observing the shape of the dataframe
                and checking any null values present.''')
    st.code('''
            # check data

            tweets_data.head()
            tweets_data.shape

            # Cheking for null values in the columns
            tweets_data.isnull().sum()

            # Cheking the distribution of the target column
            tweets_data['target'].value_counts()
            ''',line_numbers=True)
    st.markdown('''From the above code we can concur that the dataset has no null values, so no need to adjust the data for null values.
                Since, this is a text data with only 2 classifications (0 = Negative, 4 = Positive) we do not have any outliers.
                #
                ''')
    
    st.markdown('''
                While doing the above step let us also convert the value 4 = Positive to 1 = Positive. This helps us in directly feeding into
                the machine learning model and gets in norms with usual industry standards.
                ''')
    st.code('''
           #Coverting target value 4 to 1
            tweets_data.replace({'target': {4:1}},inplace=True)
            
            #let's check the coversion
            tweets_data['target'].value_counts()
            ''',line_numbers=True)
    
    
    st.markdown('''
                #
                Now that the target value has been converted, we can proceed with defining a function called stemming, that removes
                any unwanted data in a text like url links, @names that do not contain any useful emotion data in most of the cases.
                Doing this helps us in reducing the complexity of the model and makes the data a lot simpler.
                
                The function also removes any stopwords present in the data, along with converting the lexicons into their root words.
                We are also downloading stopwords data at the start and initialzing the lemmatizer that is responsible for the conversion
                of root words.''')
    st.code('''
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
            ''',line_numbers=True)
    
    st.markdown('''
                #
                Coming to the final step in data pre-processing we will apply the above function to all the text values in the target
                column of the dataset and create a new column to process the data into the machine learnign model.''')
    st.code('''
            # Creating new column with stemmed data

            tweets_data['stemmed_data'] = tweets_data['text'].apply(stemming)''',line_numbers=True)
    st.markdown('''#''')
    

    st.subheader('Making Train Test Split')

    st.markdown('''
                
                Now, that the pre-processing of the data is done. We can now proceed to splitting the data into training data and testing
                data. These training and testing data are then converted in arrays for the machine learning model to undestand them.''')
    
    st.markdown('''
                
                Before doing the above, let us first confirm the distribution of the target values in the dataset. Having a even distribution
                contributes to better accuracy. So, let's make sure that the data is evenly distributed''')
    st.code('''
            # Cheking the distribution of the target column

            tweets_data['target'].value_counts()
            ''',line_numbers=True)
    st.markdown('''My data shows a even distribution among the classes, so no need to further distribute the data
                #''')
    

    st.code('''
            # Splitting the data into train test split

            X = tweets_data['stemmed_data'].values
            y = tweets_data['target'].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 2)''',line_numbers=True)
    
    st.markdown('''
                In the above line of code we have taken the stemmed_data column from the dataset that is newly created and then proceeded
                to convert 80% of the values to training data and 20% of the values for testing data with a random_state =2 and stratify =y.
                Stratify = y makes sure that the splitting of the data is evenly distrbiuted along the y values,in this case targets mentiong 
                a random state in the code without leaving it to default will help the algorithm fetch the same type of set of values every 
                time it runs and just changes the content of those values. 
                #''')
    
    st.markdown('''
                Now let's convert the textual data into a list of arrays. For this we use vectorizers, first we will fit the vectorizer with the 
                training data and then use those numerical values to transform the test data. Vectorizer gives each word a token/numberical values
                and maps it like a indexed dictionary, Now this mapped dictionary will be used to convert the test data to arrays''')
    st.code('''
            #converting the text data into numerical data
            vectorizer = CountVectorizer()
            X_train = vectorizer.fit_transform(X_train)
            X_test = vectorizer.transform(X_test)''',line_numbers=True)
    
    st.markdown('''#''')

    st.subheader("Training the model")

    st.markdown('''
                
                Now that we have a list of arrays that can be feeded into a machine learning model to predict the emotion
                of the tweets. Let us start by creating a Logistic Regression model and feeding the training data into it.''')
    st.code('''
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)''',line_numbers=True)
    
    st.markdown('''
                #
                Now let us check the training accuracy of the model:
                ''')
    st.code('''
            # Training Accuracy
        X_train_predictions = model.predict(X_train)
        training_data_accuracy = accuracy_score(y_train, X_train_predictions)
        
        print("The training accuracy of the model is: ",training_data_accuracy)''',line_numbers=True)
    
    st.markdown('''
                #
                Let us also check the accuracy of the test data: ''')
    st.code('''
            #Accuracy score of test data
            X_test_predictions = model.predict(X_test)
            test_data_accuracy = accuracy_score(y_test,X_test_predictions)
            
            print("The Test accuracy is : ",test_data_accuracy)''',line_numbers=True)
    st.markdown('''
                
                Since, the testing acccuray is along 78% and the training accuracy is along 
                82% the model is considered not overfited and a good model.
                #''')


    st.subheader("Predictions")

    st.markdown('''
                
                Now let us create a while loop that takes input from the users then pre-process the data and 
                give us the predictions of the input text''')
    
    st.markdown('''
                #
                For this let us create a function that does this whole process and then later we can simply 
                call the function to predict the data: ''')
    st.code('''
            def predict(input_content : str):
                input_content = stemming(input_content)
                input_content = loaded_vectorizer.transform([input_content])
                predicts = (loaded_model.predict(input_content))

                if predicts == 0:
                    return 'Negative'
                else:
                    return 'Positive'
            ''',line_numbers=True)
    
    st.markdown('''
                #
                Let us create a while loop to take continous inputs from user to predict the emotion and 
                also give the user an option to stop when types in the word 'quit'
                ''')
    st.code('''
            while True:
                user_input = str(input())
                if user_input.lower() == 'quit':
                    break
                print(predict(user_input))''',line_numbers=True)
    
    st.markdown('''#''')
    st.subheader("Saving the model")

    st.markdown('''
                
                Now that the model has been successfully trained let us save it future uses. The saving process includes saving the 
                trained model and the vectorizer dictionary to run the converted text into the saved model. For this we use pickle 
                to save them and extract them for future uses.''')
    st.code('''
            import pickle

            # Saving the model            
            filename = 'Tweets_Sentiment_Analysis_Trained_Model_Logistic_Regression.sav'
            pickle.dump(model,open(filename,'wb'))

            #saving the vectorizer
            with open('Tweets_Sentiment_Analysis_count_vectorizer.pkl', 'wb') as f:
                pickle.dump(vectorizer, f)''',line_numbers=True)
    
    st.markdown('''#''')
    st.subheader("Conclusion")
    st.markdown('''
                
                The basic idea of this program is to let the machine understand and predict the emotion of a tweet.
                This model is trained using Logistic Regression with a training accuracy of 82.82% and a testing accuracy of 77.52%.
                This code is credited by [@NikhilNarsipalli - github](https://github.com/NIKHILNARSIPALLI) ''' )

