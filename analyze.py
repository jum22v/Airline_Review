import pandas as pd
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud

dataFrame = pd.read_csv('reviews.csv')

def cleanText(text):

    text = text.lower()

    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  
                           u"\U0001F300-\U0001F5FF"  
                           u"\U0001F680-\U0001F6FF"  
                           u"\U0001F1E0-\U0001F1FF"  
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    text = ''.join([char for char in text if char not in string.punctuation])

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

dataFrame['cleanedReview'] = dataFrame['Review'].apply(cleanText)

#print("Original:")
#print(dataFrame['Review'].iloc[0])
#print("\nClean Review:")
#print(dataFrame['cleanedReview'].iloc[0])

sia = SentimentIntensityAnalyzer()

dataFrame['sentimentScores'] = dataFrame['cleanedReview'].apply(lambda review: sia.polarity_scores(str(review)))
dataFrame['compoundScore'] = dataFrame['sentimentScores'].apply(lambda scoreDict: scoreDict['compound'])

def classifySentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

dataFrame['sentiment'] = dataFrame['compoundScore'].apply(classifySentiment)

#print(dataFrame[['cleanedReview', 'compoundScore', 'sentiment']].head())

sentimentCounts = dataFrame['sentiment'].value_counts()
print(sentimentCounts)

#BAR GRAPH OF REVIEWS
plt.figure(figsize=(8, 6))
sns.barplot(x=sentimentCounts.index, y=sentimentCounts.values)
plt.title('Distribution of Airline Review Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.show()

#WORD CLOUD
positiveReviews = ' '.join(dataFrame[dataFrame['sentiment'] == 'Positive']['cleanedReview'].astype(str))
negativeReviews = ' '.join(dataFrame[dataFrame['sentiment'] == 'Negative']['cleanedReview'].astype(str))

positiveWordcloud = WordCloud(width = 800, height = 400, background_color = 'white').generate(positiveReviews)
negativeWordcloud = WordCloud(width = 800, height = 400, background_color = 'black').generate(negativeReviews)

plt.figure(figsize=(10, 5))
plt.imshow(positiveWordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Positive Reviews')
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(negativeWordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Negative Reviews')
plt.show()