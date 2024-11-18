# Milestone-3-Pre-Processing
Here is a repository containing containing all steps of our preprocessing and the results of our first models. We have created two types of models - one that is a classifier for the category of the review based on number of stars (1-2 stars = negative, 3 stars = neutral, and 4-5 stars = positive) using the text of the reviews to predict the class. Our second model tries to extract and rank topics of the Spotify app that users find important (either good or bad). The model found the top 10 positive topics and top 10 negative topics (based on sentiment scores) and the negative topics were given manually assigned labels.

## Data Preprocessing for Classification (Sentiment Analysis) Model

### Rating Column
The ratings for the reviews were categorized into negative (1-2 stars), neutral (3 stars) and positive (4-5 stars).

### Review Column
The review text was tokenized using word_tokenize from nltk and then those tokens were converted to all lowercase and the punctuation was removed. The stem() function from PorterStemmer from nltk.stem was used to remove morphemes for words and only keep their stems. The stop words were then removed from the tokens. The tokens were concatenated together with spaces in between as a 'cleaned' version of the review. This was done for every review. Then, the reviews were split into lists of words and were fed into a Word2Vec model. Then the average of the word vectors from the review in the model were taken for each review and put into a 'vector' column.

## Training/Testing of Classification Model

The model type we used was a Logistic Regression model, where the feature we wanted to predict was the rating category (negative, neutral, or positive) and the feature used to predict it was the vectors created from the reviews using Word2Vec.

### Data Split
We used a 80/20 split for our train vs. test data.

### Training Error
WE NEED TO ADD THIS TO THE NOTEBOOK

### Testing Error
The accuracy for the testing data was 0.77. The precision for negative reviews was 0.72, 0.30 for neutral, and 0.82 for postive. The recall was 0.87 for negative, 0.01 for neutral, and 0.86 for positive. The f1-score was 0.79 for negative reviews, 0.03 for neutral, and 0.84 for positive.

### Fitting Curve
Based on the fitting curve that was created, the training accuracy decreased as the training dataset size increased from ~5,000 to ~50,000, while the test accuracy increased as the size increased, but slowed down after ~40,000 size of training set.

### Potential Next Models
FILL THIS OUT

## Conclusion for Classification Model
Based on the different scores for the testing data, the model is best at classifying positive reviews, then negative, and the worst at classifying neutral reviews. It is in fact very bad at classifying neutral reviews. It's possible this is because choosing 3 stars as 'neutral' might be a bit arbitrary, and it's possible people giving three stars might have both positive and negative things to say about the app, making them hard to classify.

WHAT CAN BE DONE TO IMPROVE THIS MODEL


## Data Preprocessing for Topic Ranking Model (Extracting important features for users of Spotify from reviews)
FILL OUT

### Data Preprocessing for Topic Ranking Model

### Training/Testing of Topic Ranking Model

### Fitting Curve

### Potential Next Models

## Conclusion for Feature Extraction Model


# Milestone-2-Data-Exploration-Initial-Preprocessing
Here is a repository containing containing all steps of our data exploration and initial preprocessing

## Data Preprocessing Overview 
To prepare our dataset for analysis, we will focus on encoding, categorizing, and scaling the review data effectively. Here is our approach:

## Pre-processing data
To pre-process our data, we will start by encoding the "review" column using several natural language processing tools. The tools that we are considering for encoding our textual
review data are the following:
+ **Bag of Words** creates a vocabulary of unique words across all reviews and represents each review as a vector of word counts based on this vocabulary.
This approach is straightforward, capturing the presence and frequency of words but not their contextual meaning.
+  **TF-IDF (Term Frequency-Inverse Document Frequency)** aka Word Frequency gives each word a weight based on its importance within the entire dataset.
Words that appear frequently in a specific review get higher weights, while words common across many reviews receive lower weights, helping to highlight distinctive terms.
+ **Word2Vec** converts reviews into vector representations in a way that captures semantic relationships between similar words and phrases (reviews closer to one another will also
  be closer together in the vector space). This technique allows the model to understand relationships and context.

Before doing any of this, however, we will be removing stop words from our review. Stop words are words that don't carry any semantic meaning, such as (a, an, and, I), etc. This 
will be done in order to clean up our review data. We will, however, keep certain stop words (such as not, no, very, and but) because they can be used in our case to intensify
("very good" as opposed to "good"), indiciate mixed feelings ("I liked this feature but not this one"), or invert sentiment ("no good" inverts "good"). We will also be dropping
the Reply column because over 99.6% of the data in that feature is null, as well as dropping the column "Time submitted" because the time a user submits a review is irrelevant to our model.

## General Strategy for Picking which Preprocessing Strategy.
Our plan is to encode our reviews using all three techniques, and later down the line, we are going to test our model to see which 
technique performs the best. We are going to split our data, use the same split to train three models, calculate accuracy and F1 scores, as well as using k-fold cross validation to see 
which encoding technique works best(this of course probably falls into Milestone 4 territory, but the goal here is to explain how we are going to pick which preprocessing technique
works best for us). 

## Additional Encoding
We will then encode the rating (the number of stars a user gives the app) into three categories: negative, neutral, and positive. Negative is 1-2 stars. Neutral is 3 stars, and Positive is 
4-5 stars.

For our dataset, we added an additional feature, which is the word count of each review. This feature, along with the 'Total_thumbsup' is something that we are planning on scaling:
 + To scale the word count columns, we will use a combination of robust and standard scaling, as there are outliers but the distribution for all three categories (negative, neutral, and postive) is relatively normal.
 + To scale the thumbs up column, we will use a log transformation as the distribution is heavily weighted towards zero but there are some reviews with a much higher count.



