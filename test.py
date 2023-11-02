from sklearn.pipeline import Pipeline
import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Load the dataset
Reviewdata = pd.read_csv('archive/train.csv')

# Data preprocessing
Reviewdata.drop(columns=['User_ID', 'Browser_Used', 'Device_Used'], inplace=True)

def text_clean_1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def text_clean_2(text):
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

Reviewdata['cleaned_description'] = Reviewdata.Description.apply(text_clean_1).apply(text_clean_2)

# Train the model
Independent_var = Reviewdata.cleaned_description
Dependent_var = Reviewdata.Is_Response

IV_train, IV_test, DV_train, DV_test = train_test_split(Independent_var, Dependent_var, test_size=0.1, random_state=225)

tvec = TfidfVectorizer()
clf2 = LogisticRegression(solver="lbfgs")

model = Pipeline([('vectorizer', tvec), ('classifier', clf2)])
model.fit(IV_train, DV_train)

# Define the Streamlit app
st.title('Sentiment Analysis of Reviews')
st.subheader('Select an option:')

# Display the percentage of good and bad reviews
st.write('Percentage for default:')
st.write(round(Reviewdata.Is_Response.value_counts(normalize=True) * 100, 2))
st.bar_chart(round(Reviewdata.Is_Response.value_counts(normalize=True) * 100, 2))

# Display the emoji faces
st.write('How are you feeling about the reviews?')
option = st.radio('Choose your mood:', ('Happy', 'Not Happy'))

# Perform sentiment analysis on input text
user_input = st.text_input("Enter your review:")
result = model.predict([user_input])

if result[0] == 'happy':
    st.write('😊 This seems to be a positive review!')
else:
    st.write('😠 This seems to be a negative review!')

# Get predictions from the test set
predictions = model.predict(IV_test)

# Display the percentage metrics
st.write("Accuracy: ", accuracy_score(predictions, DV_test))
st.write("Precision: ", precision_score(predictions, DV_test, average='weighted'))
st.write("Recall: ", recall_score(predictions, DV_test, average='weighted'))

# Show the list of reviews based on the chosen emoji
if option == 'Happy':
    st.write(Reviewdata[Reviewdata['Is_Response'] == 'happy']['Description'])
else:
    st.write(Reviewdata[Reviewdata['Is_Response'] == 'not happy']['Description'])