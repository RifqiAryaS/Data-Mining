import numpy as np
from sklearn import tree
import pandas as pd

# import dataset
dataset = pd.read_csv(
    "C:/Users/overm/Documents/Kuliah/Semester 4/Data Mining/Pertemuan 7/Dataset Iris.csv")      # memanggil data
# print(dataset.head())       # Melihat 5 data teratas

# print(x)

# mengubah kelas menjadi unique integer
dataset["Species"] = pd.factorize(dataset.Species)[0]

# print(dataset.head)

# menghapus kolom id
dataset = dataset.drop(labels="Id", axis=1)
# print(dataset.head)

# mengubah dataframe ke array menggunakan Numpy
dataset = dataset.to_numpy()
# print(dataset)

# membagi dataset ke training dan testing
dataTraining = np.concatenate((dataset[0:40, :], dataset[50:90, :]), axis=0)
dataTesting = np.concatenate((dataset[40:50, :], dataset[90:100, :]), axis=0)

# memecah dataset ke Input dan Label
InputTraining = dataTraining[:, 0:4]
inputTesting = dataTesting[:, 0:4]
labelTraining = dataTraining[:, 4]
labelTesting = dataTesting[:, 4]
# print(labelTraining)

# mendefinisikan Decission tree clasifier
model = tree.DecisionTreeClassifier()

# mentraining model
model = model.fit(InputTraining, labelTraining)

# memprediksi Input data testing
prediksi = model.predict(inputTesting)
# print("Label sebenarnya : ", labelTesting)
# print("Hasil prediksi : ", prediksi)

# menghitung akurasi
prediksiBenar = (prediksi == labelTesting).sum()
prediksISalah = (prediksi != labelTesting).sum()
print("Predisksi Benar : ", prediksiBenar, "data")
print("Predisksi Salah : ", prediksISalah, "data")
print("Akurasi : ", prediksiBenar/(prediksiBenar+prediksISalah) * 100, "%")
