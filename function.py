import re
import pickle
import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import csv


# global variable
global tempStoplist, model, vectorizer, textbersih # hanya deklarasi
cleantext = "(@[A-Za-z0-9_-]+)|([^A-Za-z \t\n])|(\w+:\/\/\S+)|(x[A-Za-z0-9]+)|(X[A-Za-z0-9]+)"

# function ubah cluster menjadi nama cluster
def cluster_name(cluster):
  if cluster == 0:
    return 'JK dan AI (C0)'
  elif cluster == 1:
    return 'RPL dan AI (C1)'
  elif cluster == 2:
    return 'RPL dan AI (C2)'
  elif cluster == 3:
    return 'AI (C3)'  
  elif cluster == 4:
    return 'AI dan RPL (C4)'
  elif cluster == 5:
    return 'RPL dan AI (C5)'
  elif cluster == 6:
    return 'AI dan RPL (C6)'
  elif cluster == 7:
    return 'JK (C7)'
  else:
    return 'RPL dan AI (C8)'

# function load
def setup():
  global tempStoplist, model, vectorizer
  model = pickle.load(open('uploads/kmeans_732.model','rb'))              # load K-Means model
  vectorizer = pickle.load(open('uploads/vectorizer_fit_732.model','rb')) # load TF-IDF model
  tempStoplist = []
  f = open("stopword_list_tala.txt", "r")                                 # load stopwordlist
  isi = f.read()
  for tempstp in isi.split():
    tempStoplist.append(tempstp.lower())

# function preprocessing text
def preprocessing(teks):
  tokens = []
  # normalisasi teks
  teks = re.sub(cleantext,' ',str(teks).lower()).strip()
  # spell check menggunakan nltk edit distance
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  for token in teks.split():
    #stemming
      token = stemmer.stem(token) #penggunaan sastrawi untuk stemming
    #print(token)
    #remove stopword
      if token not in tempStoplist: #mengecek apakah hasil stem merupakan stopword
    #print(token)
        tokens.append(token)
        teks = " ".join(tokens)
  return teks

# uji K-Means
def result_kmeans(teks):
  tfidf_mat = vectorizer.transform([teks])                # mengubah abtrak menjadi TF-IDF (vektor)
  X = tfidf_mat.todense()                                 # mengubah TF-IDF menjadi np array
  hasilcluster = cluster_name(int(model.predict(X)))      # clustering menggunakan K-Means
  return hasilcluster