# Milestone 4: Second Models

# NOTE: The notebooks for this Milestone should be Milestone4Classification.ipynb (for the first model, the false negatives/false positives, and fitting graph) and Milestone4.ipynb for the second model

### Conclusion section
#### What is the conclusion of your 2nd model? 


For our unsupervised model, we were able to improve our ranking system. In the previous milestone, we were able to only achieve a Kendall's Tau score of around 0.02, but we were able to 
get that all the way up to 0.2 this time before our hyperparameter tuning. While there is plenty of room for improvement, we believe that what contributed to this improved model was 
our manual labeling, as well as using fuzzy k means clustering (which basically allows for reviews to be members of multiple clusters, we felt this was appropriate because reviews 
oftentimes can be classified with different labels, so it makes sense for a review to belong to more than one cluster). Last time, we has used LDA to extract the top 10 positive/negative 
topics, came up with labels for the top 10 most frequent negative topics, and used a seperate tool to assign labels to get a ground truth ranking. This time, we decided to manually label
a small subset of our dataset (around 600), come up with labels for all of these, group similar labels/reviews together, and rank these labels/user complaints in order of decreasing 
frequency, with the label/complaint with the most user reviews being the most important/highly ranked issue. After clustering, and comparing the Kendall's tau score (metric for comparing 
rankings) we got last time to this time, we believe that doing our own manual labeling, as well as using a clustering tool that accounts for multi-cluster membership, we were able to get 
a more accurate result.

#### What can be done to possibly improve it?

For our unsupervised model, we are thinking of doing more hyperparameter tuning with the number PCA components as well as the fuzziness factor. We are also thinking of perhaps labeling 
even more data points to get a more accurate ground truth, as well as making sure    
   
# Milestone-3-Pre-Processing

## Preprocessing Updates
+ We decided against encoding our review data in three different ways, as the method we chose created a pretty accurate model
+ We decided to remove the feature we added in milestone 2, which was the word count of each review, as we don't think it would give us much information for either of the two models we decided to make

## Part 1: Supervised Learning Model Creation/Evaluation
The first model is pretty straightforward, we basically classified the reviews based on sentiment into good, positive, and neutral, and we verified our results by checking if a review that was classified as positive had 4-5 stars, checking if a review classified as neutral had 3 stars, and seeing if a review classified as negative had a 1-2 star rating.

## Part 2: Unsupervised Ranking Model Creation/Evaluation
Our second part (the ranking step) falls more into the category of unsupervised learning. What we decided to do was use Latent Dirichlet Allocation (LDA), which is a technique used for 
topic modeling to extract different topics amongst the reviews. Each topic has a corresponding sentiment score, and we got the top 10 positive topics (topics with the highest sentiment 
scores), and top 10 negative topics (topics with the lowest sentiment scores), and assigned descriptive labels to the negative topics (because we are concerned with ranking issues that 
users complain about). The topic with the lowest sentiment score was assigned the highest ranking, and the topic with the highest sentiment score among all negative topics was assigned the lowest ranking. We then used sentence transformers to get a sort of "ground truth". Sentence transformers looks through all negative reviews and assigns one of our labels to each one. We 
then counted the frequencies of the ground truth, compared it to our predicted rankings using kendall's tau, which evaluates rankings based on how similar they are. Although the accuracy 
for this particular model was slightly above random, as anything above 0 indicates a positive correlation, we plan on strengthening this in our next few models, as we will manually go 
through the dataset and manually assign labels as well as creating more labels, instead of relying on existing libraries. This is only the first model for part 2, and since it is 
unsupervised learning, it was a bit more difficult to find metrics for testing and creating labels for data.

## Data Preprocessing for Classification (Sentiment Analysis) Model

### Rating Column
The ratings for the reviews were categorized into negative (1-2 stars), neutral (3 stars) and positive (4-5 stars).

### Review Column
The review text was tokenized using word_tokenize from nltk and then those tokens were converted to all lowercase and the punctuation was removed. The stem() function from PorterStemmer from nltk.stem was used to remove morphemes for words and only keep their stems. The stop words were then removed from the tokens. The tokens were concatenated together with spaces in between as a 'cleaned' version of the review. This was done for every review. Then, the reviews were split into lists of words and were fed into a Word2Vec model. Then the average of the word vectors from the review in the model were taken for each review and put into a 'vector' column.

## Training/Testing of Classification Model

We encoded our ratings using word2vec, and we wanted to predict the sentiment (negative, neutral, or positive) and the feature used to predict it was the vectors created from the reviews 
using Word2Vec.

### Data Split
We used a 80/20 split for our train vs. test data.

### Training Error
The accuracy for the testing data was 0.81. The precision for negative reviews was 0.75, and 0.86 for positive. The recall was 0.80 for negative, and 0.82 for positive. The f1-score was 0.78 for negative reviews, and 0.84 for positive.

### Testing Error
The accuracy for the testing data was 0.77. The precision for negative reviews was 0.72, 0.30 for neutral, and 0.82 for postive. The recall was 0.87 for negative, 0.01 for neutral, and 0.86 for positive. The f1-score was 0.79 for negative reviews, 0.03 for neutral, and 0.84 for positive.

### Fitting Curve
Based on the fitting curve that was created, the training accuracy decreased as the training dataset size increased from ~5,000 to ~50,000, while the test accuracy increased as the size increased, but slowed down after ~40,000 size of training set.

### Potential Next Models
We hope to improve the accuracy of this model as getting an accurate sentiment (positive/negative/neutral) directly impacts our ability to create the ranked list of features. For future steps, we will look into using pre-trained sentiment analyzers, like ones provided by Hugging Face and test their accuracy.

## Conclusion for Classification Model
Based on the different scores for the testing data, the model is best at classifying positive reviews, then negative, and the worst at classifying neutral reviews. It is in fact very bad at classifying neutral reviews. It's possible this is because choosing 3 stars as 'neutral' might be a bit arbitrary, and it's possible people giving three stars might have both positive and negative things to say about the app, making them hard to classify. For future improvement, we will try other models besides Logistic Regression as well as pretrained sentiment analyzers.


## Topic Ranking Model (Extracting important features for users of Spotify from reviews)
The topic ranking model is an unsupervised learning model, and thus we do not have a training error currently. We evaluated the accuracy by manually creating labels for topics identified by LDA, and then using the allMini model from Hugging Face to classify reviews into one of the predefined topics. We then used the Kendall's Tau metric to get an idea of the correlation between the LDA model's ranking and allMini's ranking. This number was very low (0.02), but since it is above 0 it signifies that it is better than random. This was a very rough method for determining accuracy, but given this project it is the best we have for this milestone. Going forward, we plan to manually label reviews from all sentiments and ratings to get a comprehensive validation set that we can use to test our model, and we believe this will give us an idea of how to move forward. To improve our ranking, we will also try other methods. After labeling some data, we can transform this into a supervised machine learning problem, and we believe that fine-tuning an SVM and decision tree may help us get better results.



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



