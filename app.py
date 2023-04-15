import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, flash, render_template, jsonify, url_for
from function import preprocessing, result_kmeans, setup, cluster_name

app = Flask(__name__)

#inisialisasi objek budayaData
databasefilename = ""

# import model dan stopwordlist
setup()

# landing page atau page utama
@app.route('/')
def index():
  return render_template('index.html')


#  page tambah data
@app.route('/tambahdata')
def tambahdata():
  return render_template ('Form isi data.html')

#  page uji clustering
@app.route('/tesdata')
def tesdatapage():
  return render_template ('tesdata.html')

# mekanisme page uji clustering (backend)
@app.route('/result', methods=['POST'])
def clustering_result():
  original_text = request.form['text'] # get text input
  hasilprepo = preprocessing(original_text) # text preprocessing
  hasilcluster = result_kmeans(hasilprepo)
  return render_template ('tesdata.html', hasilkmeans=hasilcluster, abstrak=original_text, preprosesing=hasilprepo)

# page data skripsi
@app.route('/file')
def file():
    text = pd.read_csv('uploads/HASIL.csv', encoding='latin-1')
    temp = []
    for cluster in text['CLUSTERS']:
      temp.append(cluster_name(cluster)) # ubah cluster menjadi nama cluster
    text['CLUSTER'] = temp
    text = text[['NAMA', 'JUDUL', 'ABSTRAK', 'CLUSTER']]
    ta = text.to_dict(orient='records')
    print(ta)
    return render_template('tables.html', items=ta)


#page isi data


if __name__ == "__main__":
  app.run(host='0.0.0.0', debug=True)