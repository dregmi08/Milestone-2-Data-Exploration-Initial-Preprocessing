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

