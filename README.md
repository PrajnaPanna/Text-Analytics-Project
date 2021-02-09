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

Each document  in the dataset needs to be transformed  into TF or TF IDF vectors.Use a countvectorizer to create count vectors. Here the documents will be represented by the number  of times each word appears  in the documents. 
Most of the documents  have only few words in them hence most of the dimensions  in the vectors will have values set to 0.so the matrix is stored as a **_sparse matrix_** . Sparse  matrix representation  stores only the non zero values and their indexes in the vector.
The words that are irrelevant  in determining  the sentiment  of the documents like this ,is, was etc are called **_stop words_** can be removed  from the dictionary. 
Sklearn.feature_extraction.text provides a list of predefined stop words in English. 
All vectorizer classes take a list of stop words as parameters and remove the stop words while building  the dictionary  or feature set.  Many words appear  in multiple forms. If a word has similar meaning in all forms we can use only the root word as the feature using popular techniques like **stemming** and **lemmatization**.**Stemming** removes the differences between inflected  forms  of word to reduce each word to its root word  by mostly chopping  off the end of words( suffix).PorterStemmer and lansterstemmer are 2 popular algorithms  for stemming which have rules on how to chop off the words.

2. Lemmatization: This takes the morphological analysis of the words into consideration. It uses a language dictionary (i.e., English dictionary) to convert the words to the root word. 

Natural Language Toolkit (NLTK) is a very popular library in Python that has an extensive set of features for natural language processing. NLTK supports PorterStemmer, EnglishStemmer, and LancasterStemmer for stemming, while WordNet Lemmatizer for lemmatization.

We need to create a utility method, which takes documents, tokenizes it to create words, stems the words and remove the stop words before returning the final set of words for creating vectors.

Count Vectorizer takes a custom analyzer for stemming and stop word removal, before creating count vectors.


NAÏVE-BAYES MODEL FOR SENTIMENT CLASSIFICATION We will build a Naïve-Bayes model to classify sentiments. Naïve-Bayes classifier is widely used in Natural Language Processing and proved to give better results. It works on the concept of Bayes' theorem.
The posterior probability of the sentiment is computed from the prior probabilities of all the words it contains. The assumption is that the occurrences of the words in a document are considered independent and they do not influence each other.

sklearn.naive_bayes provides a class BernoulliNB which is a Naïve-Bayes classifier for multivariate Bernoulli models. BernoulliNB is designed for Binary/Boolean features (feature is either present or absent), which is the case here.
The steps involved in using Naïve-Bayes Model for sentiment classification are as follows: 1. Split dataset into train and validation sets. 2. Build the Naïve-Bayes model. 3. Find model accuracy.
Confusion matrix of sentiment classification model. In the confusion matrix, the rows represent the actual number positive and negative documents in the test set, whereas the columns represent what the model has predicted. 

 TF-IDF VECTORIZER TfidfVectorizer is used to create both TF Vectorizer and TF-IDF Vectorizer. It takes a parameter use_idf (default True) to create TF-IDF vectors. If use_idf set to False, it will create only TF vectors and if it is set to True, it will create TF-IDF vectors. 

TF-IDF are continuous values and these continuous values associated with each class can be assumed to be distributed according to Gaussian distribution. So, Gaussian Naïve-Bayes can be used to classify these documents. We will use GaussianNB, which implements the Gaussian Naïve-Bayes algorithm for classification.

The language people use on social media may be a mix of languages or emoticons. The training data also needs to contain similar examples for learning. Bag-of-words model completely ignores the structure of the sentence or sequence of words in the sentence. This can be overcome to a certain extent by using n-grams.

 n-gram is a contiguous sequence of n words. When two consecutive words are treated as one feature, it is called bigram; three consecutive words are called trigram and so on.




 
 
 

