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
