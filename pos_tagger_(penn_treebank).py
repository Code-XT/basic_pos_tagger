import nltk
import pickle
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('treebank')

from nltk.corpus import treebank
from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger
from nltk.tokenize import word_tokenize, sent_tokenize

# Load the Penn Treebank corpus
corpus = treebank.tagged_sents()

"""**Split the data**"""

train_size = int(0.8 * len(corpus))
train_sents = corpus[:train_size]
test_sents = corpus[train_size:]

"""**Training the taggers**"""

# Create taggers
default_tagger = DefaultTagger('NN')
unigram_tagger = UnigramTagger(train_sents, backoff=default_tagger)
bigram_tagger = BigramTagger(train_sents, backoff=unigram_tagger)
trigram_tagger = TrigramTagger(train_sents, backoff=bigram_tagger)

"""**Evaluation**"""

# Evaluate the taggers
print(f"Default Tagger Accuracy: {default_tagger.evaluate(test_sents):.2%}")
print(f"Unigram Tagger Accuracy: {unigram_tagger.evaluate(test_sents):.2%}")
print(f"Bigram Tagger Accuracy: {bigram_tagger.evaluate(test_sents):.2%}")
print(f"Trigram Tagger Accuracy: {trigram_tagger.evaluate(test_sents):.2%}")

def tag_sentence(sentence):
    # Tokenize the sentence
    tokens = word_tokenize(sentence)

    # Tag the tokens
    tags = trigram_tagger.tag(tokens)

    return tags

sentence = "Let's pen it down for future reference."
tags = tag_sentence(sentence)
print(tags)

with open('C:\My Folder\Programing\pos_tagger\pos_tagger_(penn_treebank).py.pickle', 'wb') as f:
    pickle.dump(trigram_tagger, f)