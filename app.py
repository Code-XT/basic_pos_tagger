from flask import Flask, render_template, request
app = Flask(__name__)

import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.tag import TrigramTagger

nltk.download('averaged_perceptron_tagger')

# Load the trained tagger
with open('C:\My Folder\Programing\pos_tagger\pos_tagger_(penn_treebank).py.pickle', 'rb') as f:
    tagger = pickle.load(f)  # Replace with the path to your trained tagger

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        sentence = request.form['sentence']
        tokens = word_tokenize(sentence)
        tags = tagger.tag(tokens)
        return render_template('index.html', tags=tags)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
