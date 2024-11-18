# Milestone-3-Pre-Processing
Here is a repository containing containing all steps of our preprocessing and the results of our first models. We have created two types of models - one that is a classifier for the category of the review based on number of stars (1-2 stars = negative, 3 stars = neutral, and 4-5 stars = positive) using the text of the reviews to predict the class. Our second model tries to extract features of the Spotify app that users find important (either good or bad).

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


## Data Preprocessing for Feature Extraction Model (Extracting important features for users of Spotify from reviews)
FILL OUT

### Data Preprocessing for Feature Extraction Model

### Training/Testing of Feature Extraction Model

### Fitting Curve

### Potential Next Models

## Conclusion for Feature Extraction Model



