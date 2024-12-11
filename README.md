# Milestone 5: Final Report

## Introduction
In today's competitive landscape, understanding user feedback is critical for businesses striving to enhance customer satisfaction and drive meaningful product improvements. Our project 
focuses on classifying user reviews based on their sentiment and identifying the key features most frequently highlighted in negative reviews and ranking them from most important to least 
important to fix/address. By leveraging sentiment analysis techniques and clustering methods, this project provides a systematic approach to pinpoint areas where users see the greatest 
need for improvement. Our methodology not only streamlines the review analysis process but also allows for businesses to prioritize enhancements efficiently, replacing the labor-intensive 
process of manually sifting through thousands of reviews in favor of machine learning algorithms and techniques. 

## Methods
### Data Exploration
To begin exploring the dataset of Spotify app reviews, we first examined its basic structure and features. The dataset consists of 61,594 observations and five features: Rating (the number 
of stars, from 1 to 5, given by users), Review (the textual content of user comments), Total_thumbsup (the number of thumbs up a review received), Time_submitted (the timestamp of when the 
review was submitted), and Reply (textual responses from other users or the Spotify team). We analyzed the distribution of thumbs up by star rating, the frequency of different ratings, and 
the word count of reviews by rating to uncover patterns that might inform preprocessing steps. For example, this analysis helped identify whether longer reviews tended to correlate with 
lower ratings (1-2 stars). Since one of our models focuses on feature ranking for negative reviews, we paid particular attention to the balance of negative versus positive reviews, the 
level of engagement (thumbs up) on negative reviews compared to positive ones, and any trends linking word count with sentiment ratings. Additionally, we performed a preliminary keyword 
analysis by extracting the top 20 words for each rating category (1-5 stars) to identify key themes and patterns across user feedback. These findings provided initial insights into user 
behavior and engagement, which guided our next steps in preprocessing and modeling.

### Preprocessing 
For our preprocessing, we began by dropping the Reply and Time_submitted columns. The Reply column was excluded because over 99.6% of its values were null, indicating that the majority of reviews did not receive a response. Retaining this column would introduce unnecessary sparsity and could negatively impact model performance. Similarly, we deemed the Time_submitted column 
irrelevant to the objectives of either model, as the timing of a review’s submission does not contribute meaningful insights into its content or sentiment. Next, we focused on the Review 
column and tokenized the text using the word_tokenize function from NLTK to break each review into individual words or tokens. Tokenization enabled us to analyze and process words 
individually, which is essential for capturing semantic meaning. To ensure consistency, we converted all tokens to lowercase and removed punctuation, reducing variability in the dataset 
caused by capitalization or special characters.

To further refine the text, we applied stemming using the PorterStemmer function from the nltk.stem library. Stemming simplified words to their root forms (e.g., "running" to "run"), 
ensuring that semantically identical terms were treated equivalently. While stemming can sometimes lead to minor losses in nuance, such as distinguishing between "running" (a verb) and 
"runner" (a noun), it significantly reduced dimensionality and improved computational efficiency. Additionally, we removed stop words—common terms such as "she," "and," "but," and "I" that 
carry little semantic value—using NLTK's stop word list. This step focused our analysis on the words that truly contributed to the essence of each review. Finally, the processed tokens 
were rejoined with spaces to reconstruct each review as a single, cleaned string.

These preprocessing steps were applied consistently across both models. However, for Model 2, which focused exclusively on feature ranking for negative reviews (ratings of 1-2 stars), 
preprocessing was limited to this subset of the data. By applying the same cleaning methodology to both models, we ensured clarity, consistency, and reproducibility, even though they were 
implemented in separate notebooks. Preprocessing decisions were guided by the need to reduce noise, standardize input data, and prepare text for downstream tasks such as feature extraction 
and sentiment analysis.

### Sentiment Classification: Model 1
The sentiment classifier was designed to categorize reviews as either positive or negative based on their numeric rating. Reviews with ratings below 3 were labeled as negative, while those with ratings of 3 or 
higher were labeled positive. We trained a  Word2Vec model on the tokenized reviews to learn vector representations for each word, capturing the semantic relationships between words in a 100-dimensional space. 
To represent entire reviews numerically, the average Word2Vec vector of all words in a review was calculated. Reviews with no valid words were represented as zero vectors. These averaged vectors were then used 
as the features for the model. A Logistic Regression classifier was used to classify the sentiment of the reviews. After splitting the data into training and test sets (80% and 20%, respectively), the Logistic 
Regression model was fitted to predict whether a review is likely to correspond to a positive or negative rating.

### Feature Ranking: Model 1
For the feature ranking model, Latent Dirichlet Allocation (LDA) was used for topic modeling to identify distinct topics within the dataset of reviews. LDA is an unsupervised learning technique that assumes each 
document (in this case, a review) is a mixture of topics, and each topic is represented by a distribution of words. The topics were extracted by finding clusters of words that frequently appeared together, 
representing common themes in the reviews. Once the topics were identified, sentiment scores were calculated for each topic based on the sentiment of the reviews associated with them. Sentiment scores were 
assigned by mapping the predicted sentiment labels (positive or negative) to numerical values: 1 for positive and -1 for negative. The model then aggregated these sentiment scores across all reviews linked to 
each topic, resulting in a cumulative sentiment score for each topic. These scores were used to rank the topics from the most negative to the most positive. Only the negative topics were considered for ranking, 
as the goal was to focus on the issues that users were most dissatisfied with. After generating the topic rankings, the words associated with each topic (provided by LDA) were manually examined to assign labels 
that more clearly described the underlying issues. For example, a topic with words like ["ads", "30 sec", "annoying"] would be labeled as "too many ads." This manual labeling resulted in a final ranked list of 
issues, replacing the original topic vectors with understandable, descriptive labels. To validate the predicted rankings and create a ground truth, Sentence Transformers were used to classify a subset of 
negative reviews according to the predefined labels derived from the LDA topics. The frequency of each label was counted, and the issues were ranked according to their frequency. This ground truth ranking was 
then compared to the predicted rankings using Kendall’s Tau, a metric that measures the correlation between two rankings. This comparison helped assess how well the model’s predicted rankings aligned with the 
actual distribution of user complaints across the topics.







