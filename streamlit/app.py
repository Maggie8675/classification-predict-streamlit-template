# libraries used for text cleaning and preprocessing
import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, TreebankWordTokenizer
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import wordnet
from collections import Counter
import html
import os
import string
import re

# libraries used for visualization and exploratory data analysis
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# libraries used for text engineering and model training and analysis
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, log_loss, f1_score
import scikitplot
from scikitplot.metrics import plot_roc, plot_confusion_matrix

# import the natural language tools we will use for text cleaning and processing
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

# Load your raw data

raw = pd.read_csv("resources/train.csv")

train = raw.copy()

# The main function where we will build the actual app
def main():
    st.sidebar.image("img/logo.jpg", use_column_width=True)  # Replace "path_to_your_logo.png" with the actual path to your logo image
    st.title("Highfliers Solutions")
    
    
    # Create a sidebar menu
    page = st.sidebar.selectbox("Select a page", ("Home", "EDA", "Model", "Text_Classification"))
    
    # Render the selected page
    if page == "Home":
        render_home_page()
    elif page == "User_Input":
        render_user_input_page()
    elif page == "EDA":
        render_eda_page()
    elif page == "Model":
        render_model_page()
    elif page == "Text_Classification":
        render_Text_Classification_page()
    elif page == "EDA":
        render_eda_page()

# create function that will clean tweets
def clean_text(text):

    """"

    The purpose of this function is to remove symbols and punctuations which don't contribute towards any text analysis
    as they don't provide any sentiment or emotion. 
    They are specifically used for grammatical purposes or domain addresses.

    """

    # remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # remove @ mentions
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    # remove hashtags
    text = re.sub(r'#', '', text)
    # remove punctuations 
    text = re.sub('[^a-zA-Z]', ' ', text)
    # remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # removing RT
    text = text.replace("RT", "")
    # remove whitespaces
    text = text.strip()

    return text

# function that will transform raw text into useful tokens
def tokenize(text):
    # tokenize text
    tokens = word_tokenize(text)
    # remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # perform stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(w) for w in tokens]
    # perform lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return tokens

def render_home_page():
    # Home page content
    st.markdown("## About Highfliers Solutions")
    st.markdown("We are a renowned data science and analytics company that focuses on utilizing advanced technologies and machine learning to derive valuable insights from extensive data sets. Our primary objective is to assist companies in developing effective marketing strategies and campaigns, with the ultimate aim of fostering a greener and more sustainable world.")

    st.markdown("At Highfliers Solutions, our mission is to harness the power of data-driven research and analytics to empower businesses in making informed decisions.")

    st.markdown("Our vision is to drive positive change by leveraging advanced technologies and providing innovative solutions that contribute to a sustainable and environmentally conscious world.")

    st.markdown("## Services we offer")
    st.markdown("We offer a wide range of services, including data analytics, machine learning, web development, and many other services tailored to meet the unique needs of our clients.")

    st.markdown("## Highfliers Solutions Team")
    st.markdown("Our team consists of highly skilled data scientists, engineers, and industry experts who are passionate about making a difference and dedicated to delivering exceptional results.")
    
    st.markdown("## Defining Climate Change")
    st.markdown("""
    Climate change has emerged as one of the most polarizing topics of our time, drawing attention from all sectors of society. According to the United Nations, climate
    change refers to long-term shifts in temperatures and weather patterns, primarily
    driven by human activities like burning fossil fuels since the 1800s. We now face
    detrimental consequences such as floods, droughts, wildfires, deforestation, and
    ecosystem loss, which have raised concerns among governments, businesses, non-
    profit organizations, and academics. Understanding the varied sentiments and

    opinions about climate change presents a significant challenge as we strive to grasp
    its implications and take appropriate actions to mitigate its impact. However, not
    everyone shares the same belief in climate change or its impact on society.
    """)

    # Contact Us section
    st.markdown("## Contact Us")

    # Business address
    st.markdown("Highfliers Solutions")
    st.markdown("123 Main Street")
    st.markdown("City, State, Country")
    st.markdown("Postal Code")

    # Map display
    st.markdown("### Map")
    map_url = "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3310.2060724290072!2d18.453945075598234!3d-33.93582757320105!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x1e950dfba71a8943%3A0xde990ec52bd6b775!2sExploreAI%20Academy!5e0!3m2!1sen!2sza!4v1687947581213!5m2!1sen!2sza"
    st.components.v1.iframe(map_url, height=500)

    # Contact form
    st.markdown("### Contact Form")
    with st.form("contact_form"):
        # Name input field
        name = st.text_input("Name")

        # Email input field
        email = st.text_input("Email")

        # Message textarea
        message = st.text_area("Message")

        # Submit button
        submit_button = st.form_submit_button("Submit")

    # Handle form submission
    if submit_button:
        # Perform necessary actions with the form data (e.g., send an email)
        # You can access the values entered in the form fields (name, email, message) as variables (name, email, message)
        # Example: Send an email using the form data
        st.success("Thank you for your message! We will get back to you soon.")

def render_eda_page():
    st.header("EDA page")

    # Overview of the EDA 
    st.subheader("Overview")
    eda_text = "This section shows some graphical illustrations of the statistical distribution of tweets \
    in the dataset used in training the Machine Learning Models.\
    \n\nThree major plots were used during the exploratory data analysis of the train data set, namely:\
    \n1. The distribution of the data set.\
    \n2. The length of Tweets for each category\
    \n3. And the top hashtags for each category\
    \n\n This will later be seen to be an insight into how convinced each category of the people are about their opinions."
    st.markdown(eda_text)
    
    # Tweet Class Imbalance
    st.subheader("Tweet Class Imbalance")
    text2 = "Class imbalance refers to a situation whereby the distribution of classes is unbalanced. One class in the dataset (i.e. the majority class) is more represented in the dataset than another class (i.e the minority). The main consequence of class imbalance is that models trained on such data will be more accurate in it's predictions on the majority class and noticeably less accurate on the minority class. \n\nSome of the causes of class imbalances include (but not limited to):\n1. The nature of the subject matter of the data may lead to certain classes naturally being more common\n2. Certain data may be scarce and difficult to collect.\n3. Some of the data from classes might be missing."
    st.markdown(text2)
    
    text1 = "The image below shows a plot shows how the tweets are distributed across each class."
    st.markdown(text1)
    tweet_dist_image = Image.open('resources/class_imbalance.png')
    st.image(image=tweet_dist_image, caption="class imbalance")
    text3 = "\n1. The data from the message column is imbalanced towards positive sentiment. \n2. Negative sentiment is poorly represented in this dataset, sharing only 8% of the total sentiment. \n3. Modelling on this dataset without balancing it will lead to well predicted positive sentiment but potentially weak predictions on the other classes due to limited data to train on and extract patterns. \n4. Another explanation could be that people who believe in climate change tend to express their beliefs on the topic more on twitter than people who don't believe in it, therefore it is easier to find and collect data from the positive sentiment class."
    st.markdown(text3)
    
    
    
    # Length of Tweet per class
    st.subheader("Tweet per Class")
    text = "Analyzing the length of a tweet itself may not provide much information for sentiment analysis.\
    \n\nHowever, in consideration with other analyses, it can provide some insight. Longer tweets would normally contain more narration and expression, and therefore provide more easily identifiable sentiment than shorter tweets.\
    \n\nThis could be useful in distinguishing between neutral tweets and news tweets which are factual or have little/no opinion vs emotive tweets which contain more tone, expression and opinion."
    st.markdown(text)
    
    length_of_tweet_per_class_image = Image.open('resources/tweet_length.png')
    st.image(image=length_of_tweet_per_class_image, caption="Most of the tweets in the dataset have between approximately 70 - 120 words. Of course, because this is assessed on the clean tweets - the actual number of characters per tweet are higher. We specifically wanted to count the words.")
    
    
    # text length(box plot)    
    st.subheader("Length of tweets")
    length_of_tweet_image = Image.open('resources/box_plot.png')
    st.image(image=length_of_tweet_image, caption="Box plot for Length of tweet per class.")
    
    text4 = "\n\nObservations about the text length of tweets and the above box plot:\
    \n1. The above box plot indicates the distribution of the length of tweets per sentiment class. \n2. The Neutral sentiment class has the lowest length of tweets with an average length of approximately 83 words. \n3. The Positive and Negative classes have the highest average length of tweets of approximately 100 and 98 words, respectively. \n4.The News class has the shortest number of average tweets (approximately 78 words) with the lowest minimum and maximum lengths as indicated by the outline range of the box - this could be because factual tweets use less adjectives and narrative language. \n5. The longer tweets have more description and narration, therefore may contain more sentiment. \n6. The neutral and news sentiment class have little/no outliers."
    st.text(text4)
    
    # Top hashtags
       
    st.subheader("Top Hashtags")
    text1 = "Hashtags being a powerful feature used in sorting and organizing tweets \
     provide an excellent approach to show that a content is related to a specific issue.\
     \n\nIt could be helpful in unraveling what the most popular hashtags are in each of the classes which would\
    help in obtaining a better grasp of the types of knowledge ingested and shared by members of each class."
    st.markdown(text1)

    st.text("\n\n\n\n\n")
                        
    # Graph 1 for hashtags
    image_text_1 = "Top Hashtags for on Climate Change"
    st.markdown(image_text_1)
    top_image = Image.open('resources/top_#.png')
    st.image(image=top_image, caption="Top Hashtags for on Climate Change")

    st.text("\n\n\n\n\n")
    

def render_user_input_page():
    st.header("User_Input Page")
    
    # Building and displaying the Exploratory data analysis page
    st.title("Exploratory Data Analysis")

    # Selecting which graphical illlustration to view
    eda_option = ["Overview" , "Tweet Distribution", "Length of tweet per class", "Top hastags per class", "Tweet Class Imbalance"]
    eda_labels = "Select a graphical distribution of tweets to view."

    eda_selection = st.sidebar.radio(eda_labels, eda_option)

    # Overview of the EDA
    
    st.subheader("Overview")
    eda_text = "This section shows some graphical illustrations of the statistical distribution of tweets \
    in the dataset used in training the Machine Learning Models.\
    \n\nThree major plots were used during the exploratory data analysis of the train data set, namely:\
    \n1. The distribution of the data set.\
    \n2. The length of Tweets for each category\
    \n3. And the top hashtags for each category\
    \n\n This will later be seen to be an insight into how convinced each category of the people are about their opinions."
    st.markdown(eda_text)

    # File upload widget
    st.subheader("Loading The Data")
    st.image("img/twit5.gif", use_column_width=True)
    train_file = st.file_uploader("Upload train set (CSV file)", type="csv")
    test_file = st.file_uploader("Upload test set (CSV file)", type="csv")

    # Perform actions after files are uploaded
    if train_file and test_file:
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        
     

        # create a copy of train data for eda
        eda = train.copy()

        # dataset for modelling purposes
        train_model = train.copy()
        test_model = test.copy()

        # Display summary of train dataset
        st.subheader("Train Dataset Summary")
        st.write(train.head(5))  # Displaying the first 5 rows of the train dataset
        
        # Display summary of test dataset
        st.subheader("Test Dataset Summary")
        st.write(test.head(5))  # Displaying the first 5 rows of the test dataset

        # Perform additional EDA operations on the datasets and display the results
        
        # Example: Displaying the shape of the train and test datasets
        st.subheader("Dataset Shapes")
        st.write("Train dataset shape:", train.shape)
        st.write("Test dataset shape:", test.shape)
        # Perform your EDA operations here
        # You can access the uploaded train and test data as DataFrames: train_df, test_df
        # Display the EDA results using Streamlit widgets or write them to the page directly
        # Example: st.write(train_df.head())
        # create expression to extract the hashtag symbol
        pattern = r'\#(\w+)'
        eda['hashtags'] = eda['message'].str.extract(pattern, expand=False)
        eda['message'] = eda['message'].apply(clean_text)
        eda['message'].head()
        # check for null/empty cells in the eda dataset
        eda.isnull().sum()
        # check the unique values of the sentiment column
        eda['sentiment'].value_counts()

        st.subheader("Text Preprocessing")
        st.image("img/Structured.jpg", use_column_width=True)

        # Import the necessary libraries
        # Create separate dataframes for each class
        news = eda[eda['sentiment'] == 2]
        positive = eda[eda['sentiment'] == 1]
        neutral = eda[eda['sentiment'] == 0]
        negative = eda[eda['sentiment'] == -1]

        # Function that will return the share of each class
        def class_share(classes, df):
            share = round((len(classes) / len(df)) * 100, 0)
            return share

        # Calculate the share of each class
        news_share = class_share(news, eda)
        positive_share = class_share(positive, eda)
        neutral_share = class_share(neutral, eda)
        negative_share = class_share(negative, eda)
        class_share_list = [news_share, positive_share, neutral_share, negative_share]

        # Plot class imbalance
        # Assign labels for the pie chart
        labels = ['news', 'positive', 'neutral', 'negative']
        sizes = class_share_list  # Percentages for each slice

        # Pie chart
        fig1, ax1 = plt.subplots(figsize=(8, 8))  # Set the figure size
        sns.set_palette('colorblind')  # Set color palette
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)  # Set parameters that ensure % shares are shown in the chart

        # Edit pie chart
        st.subheader("Share of classes in the sentiment column")

        # Display the pie chart in Streamlit
        st.pyplot(fig1)

        st.subheader("Tweet Length")
        # Calculate the length of raw tweets
        eda['text_length'] = eda['message'].apply(lambda x: len(x))
        # Creating a seaborn histogram to reflect the length of tweets in the message column
        fig2, ax2 = plt.subplots(figsize=(8, 6))  # Set the figure size
        sns.histplot(eda['text_length'], bins=5, kde=False, color='green')
        plt.xlabel('Tweet length')
        plt.ylabel('Frequency')
        plt.title('Length of words in tweets')

        # Display the bar graph in Streamlit
        st.pyplot(fig2)

        st.subheader("Word Cloud")
        # combine all the text into a single string
        text = ' '.join(eda['message'])

        # create a WordCloud object
        wordcloud = WordCloud(width=800, height=400).generate(text)

        # plot the word cloud
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        # Display the word cloud in Streamlit
        st.image(plt.gcf())  # Display the current figure as an image
        # Clear the current figure to avoid displaying it later
        plt.clf()
        
    
    
def render_model_page():
    st.header("Model Page")

    # File upload widget
    st.subheader("Loading The Data")
    train_file = st.file_uploader("Upload train set (CSV file)", type="csv")
    test_file = st.file_uploader("Upload test set (CSV file)", type="csv")

    # Perform actions after files are uploaded
    if train_file and test_file:
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)

        # create a copy of train data for eda
        eda = train.copy()

        # dataset for modelling purposes
        train_model = train.copy()
        test_model = test.copy()

        # Display summary of train dataset
        st.subheader("Train Dataset Summary")
        st.write(train.head(5))  # Displaying the first 5 rows of the train dataset
        
        # Display summary of test dataset
        st.subheader("Test Dataset Summary")
        st.write(test.head(5))  # Displaying the first 5 rows of the test dataset

        # create expression to extract the hashtag symbol
        pattern = r'\#(\w+)' 

        # extract the word from the hashtag to a new column in the train, test and eda dataframes
        train['hashtags'] = train['message'].str.extract(pattern, expand=False)
        test['hashtags'] = test['message'].str.extract(pattern, expand=False)

        # the new hashtag column 
        train['hashtags'].head(10)

        st.subheader("Modelling")
        # apply text cleaning function to the message column of train, test and eda dataframes
        train['message'] = train['message'].apply(clean_text)
        test['message'] = test['message'].apply(clean_text)

        # apply token function to message column 
        train['message'] = train['message'].apply(tokenize)
        test['message'] = test['message'].apply(tokenize)

        # assign features and label variable
        X = train['message']
        X_test = test['message']
        y = train['sentiment']

        # split dataset into training and validation datasets
        X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.1,random_state=42)

        # edit form of train, validation and test features appropriate for modelling
        X_train = list(X_train.apply(' '.join))
        X_val = list(X_val.apply(' '.join))
        X_test = list(X_test.apply(' '.join))

        st.subheader("Naive Bayes Model")
        # create vectorizer object
        vectorizer = CountVectorizer(analyzer='word')

        # fit vectorizer with train data
        vectorizer.fit(X_train)

        # transform train, test and validation features
        X_nb_train = vectorizer.transform(X_train)
        X_nb_val = vectorizer.transform(X_val)
        X_nb_test = vectorizer.transform(X_test)

        # instantiate Bayes Naive object
        model = MultinomialNB()

        # fit Bayes Naive model
        model.fit(X_nb_train, y_train)

        # make in-sample predictions on validation data
        bn_val_pred = model.predict(X_nb_val)
        # make out-of-sample predictions
        bn_pred = model.predict(X_nb_test)

        st.subheader("Logistic Regression Model")

        # create an instance of the model
        log_m = LogisticRegression(max_iter=1000)

        # create the vectorizer instance
        log_vect = CountVectorizer(analyzer='word')

        # fit features to vectorizer
        log_vect.fit(X_train)

        # transform train, validation, and test features
        X_log_train = log_vect.transform(X_train)
        X_log_val = log_vect.transform(X_val)
        X_log_test = log_vect.transform(X_test)

        # fit transformed features to model
        log_m.fit(X_log_train, y_train)

        # make in-sample prediction
        log_val_y = log_m.predict(X_log_val)

        # make out-of-sample prediction
        log_test_y = log_m.predict(X_log_test)

        # Display a progress bar
        progress_bar = st.progress(0)

        # Update progress bar
        progress_value = 0
        total_steps = 7  # Total number of steps in the code block

        with st.spinner("Running the code..."):
            for step in range(total_steps):
                # Perform each step of the code block
                # ...

                # Update progress bar after each step
                progress_value += 1
                progress_bar.progress(progress_value / total_steps)

                # Sleep for a short duration if needed between steps
                # time.sleep(1) 

        # Remove the progress bar once the code execution is complete
        progress_bar.empty()

        # Show the results or further processing
        # ...

        st.subheader("Support Vector Classification Model")

        # Load and display the images
        image1 = Image.open("img/left.jpg")
        image2 = Image.open("img/right.jpg")

        # Display the images side by side
        col1, col2 = st.columns(2)

        # Place the first image in the first column
        with col1:
            st.image(image1, use_column_width=True)

        # Place the second image in the second column
        with col2:
            st.image(image2, use_column_width=True)

        # Tranform train, val, and test data into numerical form for linear SVC modeling
        svc_vectorizer = TfidfVectorizer(ngram_range=(1,2))
        X_train_tfidf = svc_vectorizer.fit_transform(X_train)
        X_test_tfidf = svc_vectorizer.transform(X_test)
        X_val_tfidf = svc_vectorizer.transform(X_val)

        # Create SVC model object
        svc = SVC(C=1.0, kernel='linear')

        # Fit SVC model
        svc.fit(X_train_tfidf, y_train)

        # Make in-sample predictions using validation data
        svc_val_pred = svc.predict(X_val_tfidf)

        # Make out-of-sample predictions using test data
        svc_pred = svc.predict(X_test_tfidf)

        # Display a progress bar
        progress_bar = st.progress(0)

        # Update progress bar
        progress_value = 0
        total_steps = 7  # Total number of steps in the code block

        with st.spinner("Running the code..."):
            for step in range(total_steps):
                # Perform each step of the code block
                # ...

                # Update progress bar after each step
                progress_value += 1
                progress_bar.progress(progress_value / total_steps)

                # Sleep for a short duration if needed between steps
                # time.sleep(1) 

        # Remove the progress bar once the code execution is complete
        progress_bar.empty()

        # Show the results or further processing
        # ... 

        st.subheader("Model Performance")
        st.image("img/model.gif", use_column_width=True)

        st.subheader("Naive Bayes Model")
        print("Classification Report for Naive Bayes Model:\n",classification_report(y_val, bn_val_pred))
        # plot confusion matrix for Naive Bayes model
        figNaive, axN = plt.subplots()
        confusion_matrix = plot_confusion_matrix(y_val, bn_val_pred, normalize=True, cmap='coolwarm', ax=axN)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Display the confusion matrix plot in Streamlit
        st.pyplot(figNaive, figsize=(8, 8))

        # Calculate the classification report
        nb_report = classification_report(y_val, bn_val_pred, output_dict=True)
        nb_df = pd.DataFrame(nb_report).transpose()
        nb_df.drop(['accuracy'], inplace=True)
        nb_df.sort_values(by=['f1-score'], ascending=True, inplace=True)
        sentiment_classes = nb_df.drop(['weighted avg', 'macro avg'])
        sentiment_classes['f1-score'].plot(kind='bar', figsize=(5, 5))
        plt.xlabel('f1-score')
        plt.ylabel('Sentiment Class')
        plt.title('Naive Bayes Model: F1 score per sentiment class')

        # Display the bar plot in Streamlit
        st.pyplot(plt.gcf())

        st.subheader("Logistic Regression Model")
        # summary of classification metrics assessing performance of Logistic Regression Model
        print("Classification Report for Logistic Regression Model:\n",classification_report(y_val, log_val_y))
        # plot confusion matrix for Logistic Regression model
        figLogistic, axL = plt.subplots()
        confusion_matrix2 = plot_confusion_matrix(y_val, log_val_y, normalize=True,figsize=(8,8),cmap='BuGn', ax=axL)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Display the confusion matrix plot in Streamlit
        st.pyplot(figLogistic, figsize=(8, 8))
        
        st.subheader("Linear SVC Model")
        #summary of classification metrics assessing performance of Linear SVC model
        print('Classification Report for Linear SVC Model:\n', classification_report(y_val, svc_val_pred))
        # plot confusion matrix for Linear SVC model
        figSVC, axSVC = plt.subplots()
        confusion_matrix3 = plot_confusion_matrix(y_val, log_val_y, normalize=True,figsize=(8,8),cmap='BuGn', ax=axSVC)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Display the confusion matrix plot in Streamlit
        st.pyplot(figSVC, figsize=(8, 8))

        st.subheader("Modelling For Tweet Sentiment Analysis")
        
        
def render_Text_Classification_page():
    st.header("Text_Classification Page")
        
    # File upload widget
    st.title("Tweet Classification using Machine Learning Models")
    #st.subheader("Climate change tweet classification")
    #st.subheader("Modelling")
    st.info("Prediction with ML Models")
    # Creating a text box for user input
    tweet_text = st.text_area("Enter Text","Type Here")
    
    # cleaning the input text before passing it into the machine learning model for prediction
    tweet_text = tweet_text.replace('\n', '')
    tweet_text = html.unescape(tweet_text)
    
    # removing special characters from text
    tweet_text = re.sub(r"(@[A-Za-z0-9_]+)|[^\w\s]|http\S+", "", tweet_text)
    tweet_text = tweet_text.lower()# converts text to lowercase
    tokeniser = TreebankWordTokenizer() # creating an instance of the TreebankWordTokenizer
    tweet_text_token = tokeniser.tokenize(tweet_text)# transforming input into tokens
    
    #This function obtains a pos tag and returns the part of speech.
        #Input:
        #tag: POS tag
        #datatype: str
        #Output:
        #wordnet.pos: Part of Speech
        #datatype: str
        
    def get_pos(tag):
        if tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('R'):
            return wordnet.ADV
        elif tag.startswith('N'):
            return wordnet.NOUN
        else:
            return wordnet.NOUN
    
    pos_tag = PerceptronTagger()
    tweet_text_POS_tag = pos_tag.tag(tweet_text_token)# gets the part of speech tag of each word in the sentence
    tweet_text_POS_tag = [(word, get_pos(tag)) for (word, tag) in tweet_text_POS_tag]
    lemmatizer = WordNetLemmatizer()#creating an instance of the lemmatizer
    tweet_text_token = [lemmatizer.lemmatize(token) for token in tweet_text_token]# gets the lemma of each word
    tweet_text_token =  [word for word in tweet_text_token if not word in stopwords.words('english') and word != 'not']
    tweet_text_token = ' '.join(tweet_text_token)
    # A list of available models that can be used for the classification
    model_option = ["Logistic Regression", "Linear Support Vector Classifier", "Naive Bayes"]
        
    model_selection = st.selectbox("Select a model type you will like to use as the classifier.", model_option)
  
    # Selecting from multiple models
    # If Logistic Regression is selected
    if model_selection == "Logistic Regression":
        predictor = joblib.load(open(os.path.join("resources/lg.pkl"),"rb"))
    # If Linear Support Vector Classifier is selected
    if model_selection == "Linear Support Vector Classifier":
        predictor = joblib.load(open(os.path.join("resources/svc.pkl"),"rb"))
    # If Naive Bayes is selected
    if model_selection == "Naive Bayes":
        predictor = joblib.load(open(os.path.join("resources/nb.pkl"),"rb"))
        
    if st.button("Classify"):
        # Transforming user input with vectorizer
        vect_text = [tweet_text]
        
        # Load your .pkl file with the model of your choice + make predictions
        # Try loading in multiple models to give the user a choice
        prediction = predictor.predict([tweet_text])
        # When model has successfully run, will print prediction
        # Creating a dictionary to show a more human interpretation of result
        result_dict = {'News': 'This tweet links to factual news about climate change.',
                        'Pro': 'This tweet supports the belief of man-made climate change.',
                        'Neutral': 'This tweet is neutral.',
                        'Anti': 'This tweet does not support the man-made climate change belief.'}
        
        result = result_dict[prediction[0]]
        st.success("Text Categorized as: {}".format(prediction[0]))
        st.success("{}".format(result))

    
# Required to let Streamlit instantiate our web app.    
    
if __name__ == "__main__":
    main()
