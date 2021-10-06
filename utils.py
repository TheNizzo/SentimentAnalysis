from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from typing import List
from typing import Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
import spacy
import nltk
import re
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def stem(l: List[str]) -> List[str]:
  """
  Takes in a list of phrases l, applys stemming, returns the stemmed list.
  """
  res = []
  re_word = re.compile(r"^\w+$")
  stemmer = SnowballStemmer("english")
  for text in tqdm(l, total=len(l)):
    res.append(" ".join([stemmer.stem(word) for word in word_tokenize(text.lower()) if re_word.match(word)]))
  return res

def lemm(l: List[str]) -> List[str]:
  """
  Takes in a list of phrases l, applys lemmatization, returns the lemmatized list.
  """
  lemmas = []
  re_word = re.compile(r"^\w+$")
  for text in tqdm(l, total=len(l)):
    lemmas.append(' '.join([token.lemma_ for token in nlp(text.lower()) if re_word.match(token.text)]))
  return lemmas


def pos(l: List[str], d: Dict[str, float]) -> int:
  """
  Takes a list of words l, and a dictionnary of rated words d, returns the number of positive words in the list.
  """
  pos = 0
  for w in l:
    if w in d and d[w] > 0.5:
      pos += 1
  return pos

def neg(l: List[str], d: Dict[str, float]) -> int:
  """
  Takes a list of words l, and a dictionnary of rated words d, returns the number of negative words in the list.
  """
  neg = 0
  for w in l:
    if w in d and d[w] < 0.5:
      neg += 1
  return neg

def contains_no(l: List[str]) -> bool:
  """
  takes in a list of words l, returns True if list contains the word "no" else False.
  """
  return 1 if ("no" in list((map(lambda x: x.lower(),l)))) else 0

def first_second_pro(l: List[str]) -> int:
  """
  takes in a list of words l, returns count of first and second pronouns in the list.
  """
  pronouns = ["i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your",
              "yours"]
  return sum([list((map(lambda x: x.lower(),l))).count(j) for j in pronouns])

def get_features(df: pd.DataFrame, d: Dict[str, float]):
  """
  takes in a dataframe df, and a dictionnary d of rated words, changes the df with added features columns.
  """
  split_df = df['val'].str.split("[ .,\"]")
  df['containsNO'] = split_df.apply(contains_no)
  df['containsExclamation'] = df['val'].apply(lambda x: 1 if "!" in x  else 0)
  df['count_pronouns'] = split_df.apply(first_second_pro)
  df["logNOfWords"] = np.log(df["val"].str.count(" "))
  df["pos_count"] = split_df.apply(pos, args=(d,))
  df["neg_count"] = split_df.apply(neg, args=(d,))
  difference = df["pos_count"] - df["neg_count"]
  df["isposgreater"] = difference.apply(lambda x: 1 if x > 0 else 0)