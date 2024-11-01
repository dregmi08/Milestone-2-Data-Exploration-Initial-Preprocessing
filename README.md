# Milestone-2-Data-Exploration-Initial-Preprocessing
Here is a repository containing containing all steps of our data exploration and initial preprocessing

## Pre-processing data
To pre-process our data, we will start by encoding the "review" column using natural language processing tools such as word2vec, bag of words, and word frequency. This will result in columns for each technique so we can then test which technique performs the best.
We will then encode the rating into three categories: negative, neutral, and positive. Negative is 1-2 stars. Neutral is 3 stars, and Positive is 4-5 stars.
We will create descriptor columns for the "review" column, such as length, word count, and number of capital letters.
To scale these descriptor columns, we will use a combination of robust and standard scaling, as there are outliers but the distribution for all three is relatively normal.
To scale the thumbs up column, we will use a log transformation as the distribution is heavily weighted towards zero but there are some reviews with a much higher count.