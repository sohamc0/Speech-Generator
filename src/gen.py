from gensim.models import Word2Vec
from tensorflow.keras import models
import numpy as np
import os

pwd = os.getcwd()
word_model = Word2Vec.load(pwd+"/../models/word2vec.model")
model = models.load_model(pwd+"/../models/speech-generator.keras")

def word2idx(word):
	return word_model.wv.get_index(word)
index_to_key_ls = word_model.wv.index_to_key
def idx2word(idx):
	return index_to_key_ls[idx]

def sample(preds, temperature=1.0):
	if temperature <= 0:
		return np.argmax(preds)
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)

def generate_next(text, num_generated=10):
  word_idxs = [word2idx(word) for word in text.lower().split()]
  for i in range(num_generated):
    prediction = model.predict(x=np.asarray([word_idxs]))
    idx = sample(prediction[-1], temperature=0.71)
    if idx != 1:
      word_idxs.append(idx)
  return ' '.join(idx2word(idx).replace("+", "") for idx in word_idxs)

while True:
	prompt = input("Enter a few words to start off your sentence...\n")
	try:
		print(prompt + "... -> " + generate_next(prompt))
	except:
		print("Something went wrong. Please try again...")