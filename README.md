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
behavior and engagement, which guided our next steps in preprocessing and modeling. All of the steps we took for data exploration, along with all plots and graphs can be seen in the [Milestone 2 Notebook](https://github.com/dregmi08/SpotifyFeatureRanking/blob/Milestone5/Milestone2.ipynb)

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
For the feature ranking model, Latent Dirichlet Allocation (LDA) was used for topic modeling to identify distinct topics within the dataset of reviews. LDA is an unsupervised learning technique that assumes 
each document (in this case, a review) is a mixture of topics, and each topic is represented by a distribution of words. The topics were extracted by finding clusters of words that frequently appeared together, 
representing common themes in the reviews. Once the topics were identified, sentiment scores were calculated for each topic based on the sentiment of the reviews associated with them. Sentiment scores were 
assigned by mapping the predicted sentiment labels (positive or negative) to numerical values: 1 for positive and -1 for negative. The model then aggregated these sentiment scores across all reviews linked to 
each topic, resulting in a cumulative sentiment score for each topic. These scores were used to rank the topics from the most negative to the most positive. Only the negative topics were considered for ranking, 
as the goal was to focus on the issues that users were most dissatisfied with. After generating the topic rankings, the words associated with each topic (provided by LDA) were manually examined to assign labels 
that more clearly described the underlying issues. For example, a topic with words like ["ads", "30 sec", "annoying"] would be labeled as "too many ads." This manual labeling resulted in a final ranked list of 
issues, replacing the original topic vectors with understandable, descriptive labels. To validate the predicted rankings and create a ground truth, Sentence Transformers were used to classify a subset of 
negative reviews according to the predefined labels derived from the LDA topics. The frequency of each label was counted, and the issues were ranked according to their frequency. This ground truth ranking was 
then compared to the predicted rankings using Kendall’s Tau, a metric that measures the correlation between two rankings. This comparison helped assess how well the model’s predicted rankings aligned with the 
actual distribution of user complaints across the topics.

### Sentiment Classification: Model 2
Our second sentiment classification model used the same embedding technique as the first model, but with a few key adjustments. Instead of using Logistic Regression, we employed a Naive Bayes Classifier. We 
also introduced a neutral category by adjusting the mapping of ratings, where 3-star ratings were now considered neutral, while ratings of 4 and 5 stars were classified as positive and 1 and 2 stars as 
negative. To further optimize the performance of the classifier, we performed hyperparameter tuning, focusing primarily on the smoothing parameter (alpha). This tuning step helped refine the model's 
classification of reviews by adjusting the level of smoothing applied to frequency estimates, ultimately improving the classifier’s accuracy in distinguishing between positive, negative, and neutral reviews.

### Feature Ranking: Model 2
For our second feature ranking model, we shifted our approach by implementing clustering. We selected K-means fuzzy clustering, as it allows reviews to be members of multiple clusters, which is crucial since 
many reviews address multiple topics. This technique enables more flexibility in capturing the diverse nature of user complaints. Next, we manually labeled a small subset of our dataset (600 reviews) and 
grouped them into topics. Afterward, we ranked these topics by their frequency, with the topic having the most frequent complaints being ranked the highest. This labeled subset served as our ground truth for 
the ranking. To evaluate the accuracy of the ranking generated by K-means fuzzy clustering, we compared it to our manually created ranking using Kendall’s Tau. Additionally, we performed hyperparameter tuning 
on the model by experimenting with a different numbers of clusters to optimize the clustering results and improve the model’s performance.

## Results

### Sentiment Classifier: Model 1

#### Results on test set

|              | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| NEGATIVE     | 0.74      | 0.80   | 0.77     | 4884    |
| POSITIVE     | 0.86      | 0.82   | 0.84     | 7435    |
| **Accuracy** |           |        | 0.81     | 12319   |
| Macro avg    | 0.80      | 0.81   | 0.81     | 12319   |
| Weighted avg | 0.81      | 0.81   | 0.81     | 12319   |

#### Results on training set

|              | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| NEGATIVE     | 0.75      | 0.80   | 0.78     | 19887   |
| POSITIVE     | 0.86      | 0.82   | 0.84     | 29388   |
| **Accuracy** |           |        | 0.81     | 49275   |
| Macro avg    | 0.81      | 0.81   | 0.81     | 49275   |
| Weighted avg | 0.82      | 0.81   | 0.82     | 49275   |

### Sentiment Classifier: Model 2

#### Results on test set

|               | precision | recall | f1-score | support |
|---------------|------------|--------|----------|---------|
| Negative      | 0.72       | 0.86   | 0.79     | 4955    |
| Neutral       | 0.22       | 0.05   | 0.08     | 1327    |
| Positive      | 0.85       | 0.86   | 0.85     | 6037    |
| accuracy      |            |        | 0.77     | 12319   |
| macro avg     | 0.60       | 0.59   | 0.57     | 12319   |
| weighted avg  | 0.73       | 0.77   | 0.74     | 12319   |

#### Results on training set 

|               | precision | recall | f1-score | support |
|---------------|------------|--------|----------|---------|
| Negative      | 0.74       | 0.90   | 0.82     | 19816   |
| Neutral       | 0.67       | 0.15   | 0.24     | 5559    |
| Positive      | 0.87       | 0.87   | 0.87     | 23900   |
| accuracy      |            |        | 0.80     | 49275   |
| macro avg     | 0.76       | 0.64   | 0.64     | 49275   |
| weighted avg  | 0.79       | 0.80   | 0.78     | 49275   |

### Results of the Feature Ranking Model, Models 1 and 2 (Unsupervised, so no fitting graph or Precision/Accuracy/Recall Metrics)

| Model Name                              | Kendall’s Tau Score |
|-----------------------------------------|---------------------|
| LDA                                     | 0.02                |
| Fuzzy K-means                           | 0.2                 |
| Fuzzy K-means (post hyperparameter tuning) | 0.466               |

## Discussion

### Sentiment Model
For our supervised model (the sentiment classification), we used a Naive Bayes Classifier to classify reviews into three sentiment categories: Positive, Negative, and Neutral. This approach differs from 
Milestone 3, where we used a Logistic Regression model to classify reviews as either Positive or Negative. By introducing the Neutral category, we noticed that our overall accuracy for both training and testing 
remained similar to the previous milestone (Milestone 3: 0.81 accuracy, Milestone 4: 0.77 for testing, 0.80 for training). The slightly lower performance in the Naive Bayes model can be attributed to the 
difficulty the classifier had in accurately classifying reviews with a 3-star rating, which was designated as Neutral. The classification report revealed very low precision, recall, and F1 scores for this 
category (precision ~ test: 0.22, train: 0.67, recall ~ train: 0.05, test: 0.15, F1 ~ test: 0.08, train: 0.2). Upon further analysis, we found that some 3-star reviews contained subtle negative sentiments, 
which may have caused the classifier to misclassify them as Neutral.When we adjusted the mapping by grouping 3-star reviews as Negative (instead of Neutral) and keeping 4- and 5-star reviews as Positive, the 
accuracy of the model increased to approximately 0.85. This improvement suggested that this new approach of classifying 3-star reviews as Negative was more accurate. Despite this improvement, we still plan to 
explore more 3-star samples to confirm whether this adjustment is the most appropriate solution. Overall, we are pleased with the Naive Bayes classifier, especially given the improvements made by adjusting the 
3-star review classification. Even with the addition of the Neutral category, the model remains comparable in performance to the earlier Logistic Regression-based model, and with further refinements, it has the 
potential to be our most accurate model yet.

### Feature Ranking Model
For our unsupervised model, we significantly improved our ranking system compared to the previous milestone. In Milestone 3, our Kendall's Tau score, which is used to compare two rankings, was around 0.02. 
However, we were able to increase this to 0.2 in Milestone 4, even before performing hyperparameter tuning. While there is still room for improvement, we believe that several factors contributed to this 
improvement. One key improvement was our manual labeling process. In the previous milestone, we used LDA (Latent Dirichlet Allocation) to extract the top 10 positive and negative topics and then manually 
labeled the top 10 most frequent negative topics. We also used a separate tool to assign labels and create a ground truth ranking. However, in Milestone 4, we decided to manually label a small subset of the 
dataset (around 600 reviews). This allowed us to assign more accurate and descriptive labels to a broader set of issues. We then grouped similar labels and reviews together and ranked them by frequency, with 
the most common complaints being ranked as the most important. Additionally, we used fuzzy k-means clustering to account for multi-cluster membership, recognizing that reviews can often belong to multiple 
categories or labels. This was a crucial step because reviews typically touch on more than one issue, making it more appropriate to allow reviews to belong to multiple clusters. By clustering reviews with 
similar labels and complaints, we were able to better capture the complexity of user feedback. After clustering the reviews and comparing the new Kendall's Tau score to the previous one, we saw a clear 
improvement. The combination of manual labeling and multi-cluster membership clustering contributed to a more accurate ranking system. Furthermore, after performing hyperparameter tuning, we increased the 
number of clusters, which led to a Kendall's Tau score of 0.466, a significant improvement over the previous results and a new personal best. This marked a substantial advancement from what we had achieved in 
Milestone 3, and we are optimistic about further improving the model with continued adjustments. 

## Conclusion
Our project successfully met our goal of understanding user feedback through sentiment analysis and featuring importance rankings. Our model is designed to offer valuable insights for businesses and app developers looking to enhance their products based on customer reviews, with a particular focus on addressing negative feedback and identifying what aspects require changes. 

We are satisfied with our progress throughout this project; however, we recognize there is still considerable room for improvement. One way would be to experiment with additional models, which could potentially 
enhance Kendall’s Tau score for feature ranking, thereby refining our understanding of which features impact user satisfaction the most. Moreover, enhancements to our sentiment classifier’s performance, 
specifically for accuracy, precision, and recall, are crucial, especially for classifications of our 3-star reviews, which have a less definite user sentiment. Additionally, having a more extensive set of 
manually set ground truths would be beneficial in having a higher confidence level in our evaluation. 

If we were to use this solution for practical purposes, other functionalities would be beneficial. First, we would transform it into a continuous learning system that would dynamically incorporate new reviews 
as they are submitted. Additionally, considering the review timestamp and whether an issue has already been resolved since the review was submitted could lead to a more accurate and up-to-date list of crucial 
issues needing attention. Overall, these enhancements would make our data more responsive to real-time data and effectively guide product development.


## Statement of Collaboration

+ **Drishti Regmi**: Wrote abstract, came up with feature ranking idea, created second models for both milestones, did the bulk of the data exploration, did the Multinomial Naive Bayes Model for Milestone 4, contributed to a lot of the READMEs, and labeled 100 data points as well as hyperparameter tuning for the second model in Milestone 4
+ **Nicolas Colebank**: Created first classification model, contributed to data preprocessing/exploration README, labeled 100 data points, and helped group labels into topics. Wrote intro, methods, and results sections for the final report.
+ **Luke Valdez**: Participated in the discussion for the preprocessing, model 1 and model 2 assignments. Worked on the READMEs for hyperparameter tuning and model fitting. Labeled 100 data points for the ground truth of models 1/2. Did hyperparameter tuning for the first model of the project. Reviewed and edited the final report and implemented and labeled figures.
+ **Ezgi Bayraktaroglu**: Participated in discussions for the milestones. Helped with the original dataset proposal. Worked on the README file for the classification model for Milestone 3. Labelled 100 negative reviews with topics for the ground truth of the unsupervised model and grouped the 600 labelled reviews into more general groups/topics with Nicolas Coleback.
+ **Rushil Chandrupatla**: Refined abstract, fully created classification model 1 for sentiment using word2vec and verified validity using rating, fully created feature model 1 for topic analysis and retrieval using LDA, labeled 100 data points to create evaluation dataset, made edits on final report.
