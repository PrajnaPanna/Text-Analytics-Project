# Text-Analytics-Project
Two projects of text analytics ,sentiment analysis :pre-processing using BoW model and analysis using Naive Bayes model.

Project 1:
Dataset contains reviews comments of movies. comments on the data set already labled as either positive or negative.The dataset contains the following 2 fields seperated by tab character;
1. Text: Actual review comment on the movie
2. sentiment: positive sentiments are labelled as 1 and negative sentiments are labled as 0.

Each record or an example in the column _text_ is called a **document**.

Text pre processing or Feature Extraction:
 in Unstructured data the features have to be extracted using some process. One way to consider each woed as a feature and find a measure to capture wether a word exists or not exists in the sentence . This is called **Bag of words**.Each sentence is called a **Document**.Collection of all sentence(record) /documents is called as **Corpus**.
 
 Bag Of words model:
 Create a dictionary of all the words used in the corps.then convert each document into a vector that represents words available in the document.
 Three ways to identify the importance of the words in a BoW model:
 1. Count Vector model.
 2. Term Frequency Vector Model
 3. Term Frequency inverse Document Frequency Model
 
 For creating the **count vector** we count the occurances of each word in the given documents.
 
**Term Frequency vector** is calculated for each document in the corpus and is the frequency of each term in the document.Term Frequency for a word is also called as **Token**

**TF-IDF** measures how important a word is to a document in the Corpus .The importance of a word (token) increases proportionately to the number of times a word appears in the document but reduces by the frequency of the word present in the corpus.


 
 
 

