from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import numpy as np
import spacy
import nltk
import re
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "ner"])

def stem(l):
  res = []
  re_word = re.compile(r"^\w+$")
  stemmer = SnowballStemmer("english")
  for text in tqdm(l, total=len(l)):
    res.append(" ".join([stemmer.stem(word) for word in word_tokenize(text.lower()) if re_word.match(word)]))
  return res

def lemm(l):
  lemmas = []
  re_word = re.compile(r"^\w+$")
  for text in tqdm(l, total=len(l)):
    lemmas.append(' '.join([token.lemma_ for token in nlp(text.lower()) if re_word.match(token.text)]))
  return lemmas


def pos(l, d):
  pos = 0
  for w in l:
    if w in d and d[w] > 0.5:
      pos += 1
  return pos

def neg(l, d):
  neg = 0
  for w in l:
    if w in d and d[w] < 0.5:
      neg += 1
  return neg

def contains_no(l):
  return 1 if ("no" in list((map(lambda x: x.lower(),l)))) else 0

def first_second_pro(l):
  pronouns = ["i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your",
              "yours"]
  return sum([list((map(lambda x: x.lower(),l))).count(j) for j in pronouns])

def get_features(df, d):
  split_df = df['val'].str.split("[ .,\"]")
  df['containsNO'] = split_df.apply(contains_no)
  df['containsExclamation'] = df['val'].apply(lambda x: 1 if "!" in x  else 0)
  df['count_pronouns'] = split_df.apply(first_second_pro)
  df["logNOfWords"] = np.log(df["val"].str.count(" "))
  df["pos_count"] = split_df.apply(pos, args=(d,))
  df["neg_count"] = split_df.apply(neg, args=(d,))