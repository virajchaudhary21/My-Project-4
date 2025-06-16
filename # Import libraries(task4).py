# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Sample dataset (simulated)
data = {
    'tweet': [
        'I love the new phone, the camera is amazing!',
        'The service was terrible, never going back!',
        'Absolutely fantastic experience!',
        'The product stopped working after a week.',
        'I am very happy with the customer support.',
        'Worst purchase ever!',
        'Highly recommend this brand!',
        'Not worth the price.',
        'Great quality and fast delivery.',
        'Very disappointed, waste of money.'
    ],
    'sentiment': [
        'positive', 'negative', 'positive', 'negative', 'positive',
        'negative', 'positive', 'negative', 'positive', 'negative'
    ]
}

df = pd.DataFrame(data)

# Display data
print(df)

# Visualize sentiment distribution
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', data=df, palette='viridis')
plt.title("Sentiment Distribution")
plt.show()

# Generate WordCloud for positive and negative
for sentiment in ['positive', 'negative']:
    text = " ".join(df[df['sentiment'] == sentiment]['tweet'])
    wordcloud = WordCloud(width=600, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'WordCloud for {sentiment} tweets')
    plt.show()

# Encode target
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['sentiment'], test_size=0.3, random_state=42)

# Vectorize text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Evaluate
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
