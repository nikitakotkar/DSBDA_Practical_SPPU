from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, WordNetLemmatizer
import numpy as np

# Sample document
text = "Natural Language Processing is a field of computer science concerned with interactions between computers and human languages."

# Tokenization (splitting text into words)
tokens = word_tokenize(text.lower())  # Lowercase for case-insensitivity

# Stop words removal (removing common words)
stop_words = stopwords.words('english')
tokens = [token for token in tokens if token not in stop_words]

# Stemming (reducing words to base form)
porter = PorterStemmer()
stemmed_tokens = [porter.stem(token) for token in tokens]


# Part-of-Speech (POS) Tagging - Example with two sentences (replace with your actual sentences)
sentence1 = "This is an NLP example."
sentence2 = "We can analyze text data."
combined_tokens = word_tokenize(sentence1) + word_tokenize(sentence2)
tagged_words = pos_tag(combined_tokens)

# Lemmatization (reducing words to dictionary form)
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

# Print results
print("Original Text:", text)
print("\nTokenized Words:", tokens)
print("\nStop Words Removed:", tokens)
print("\nStemmed Words:", stemmed_tokens)
print("\nPOS Tagging:", tagged_words)
print("\nLemmatized Words:", lemmatized_tokens)


# Function to calculate Term Frequency (TF)
def calculate_tf_idf(word, document, corpus):
  """
  Calculates TF-IDF for a word in a document relative to a corpus.

  Args:
      word: The word to calculate TF-IDF for.
      document: A list of tokens representing the document.
      corpus: A list of documents, where each document is a list of tokens.

  Returns:
      The TF-IDF value for the word in the document relative to the corpus.
  """

  # Term Frequency (TF)
  tf = document.count(word) / len(document)

  # Inverse Document Frequency (IDF) - Smoothed
  doc_count = 0
  for doc in corpus:
    if word in doc:
      doc_count += 1
  idf = 1 + np.log(len(corpus) / (doc_count + 1))

  # TF-IDF calculation
  return tf * idf

# Sample document (replace with yours)
text = "Natural Language Processing is a field..."
tokens = word_tokenize(text.lower())
stop_words = stopwords.words('english')
tokens = [token for token in tokens if token not in stop_words]

# Sample corpus (replace with yours)
corpus = [tokens, ["Machine learning", "data science"]]

# Sample word
word = "language"

# Calculate TF-IDF
tfidf = calculate_tf_idf(word, tokens, corpus)
print("\nTF-IDF for", word, ":", tfidf)






