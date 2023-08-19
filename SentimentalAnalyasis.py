#Develop a Python program to perform sentiment analysis on a collection of customer reviews. You are allowed to use pre-trained language model or a sentiment analysis library. [20]

**Example**

```
#Input
  #"The product exceeded my expectations. It's excellent!",
  #"I had a terrible experience with this company. The customer service was rude and unhelpful.",
  #"The quality of the product is average. It meets my basic requirements.",
  #"I absolutely love this product! It has improved my daily routine.",

#Output - {'positive': 2, 'negative': 1, 'neutral': 1}

#*Output is a python dictionary with keys as positive, negative & neutral and value as the count of such reviews.

#Explanation -
#The product exceeded my expectations. It's excellent! ==> positive
#I had a terrible experience with this company. The customer service was rude and unhelpful. ==> negative
#The quality of the product is average. It meets my basic requirements. ==> neutral
#I absolutely love this product! It has improved my daily routine. ==> positive
```
file_path = "input/q1/reviews.txt"


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create a SentimentIntensityAnalyzer object.
sid = SentimentIntensityAnalyzer()

# Read the reviews from a file.
with open(file_path, "r") as file:
    reviews = file.readlines()

# Calculate the sentiment score for each review.
sentiment_scores = [sid.polarity_scores(review) for review in reviews]

# Round off the value of sentiment_score.
sentiment_scores = [round(score["compound"], 1) for score in sentiment_scores]

# Print the sentiment scores.
print(sentiment_scores)

# Print the sentiment labels.
sentiment_labels = [
"negative" if score < 0 else "positive" if score > 0 else "neutral"
for score in sentiment_scores
]

print(sentiment_labels)

# Print the sentiment distribution.
sentiment_dict = {
score: sentiment_labels.count(score)
for score in sentiment_labels
}

print(sentiment_dict)

