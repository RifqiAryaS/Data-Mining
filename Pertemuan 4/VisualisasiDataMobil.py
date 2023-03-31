import numpy as np                  # Numpy digunakan untuk komputasi matriks
# Matplotlib merupakan library python untuk presentasi data berupa grafik atau plot
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(
    'C:/Users/overm/Documents/Kuliah/Semester 4/Data Mining/Pertemuan 4/car2.csv')
print(dataset.head())     # Menampilkan sample data
print(dataset.info())     # Ringkasan data

# Menampilkan data secara unique pada kolom tertentu
print(dataset['doors'].unique())

# Menampilkan data pada kolom tertentu dengan isi tertentu
pintu = '3'
print(dataset[dataset['doors'] == pintu])

# Mensortir data yang akan ditampilkan
bagasi = dataset['lug_boot'].unique()
bagasi.sort()
print(bagasi)

# Menampilkan data dengan 2 filter
mobil_bagus = dataset[dataset['doors'] ==
                      '4'][dataset['persons'] == 'more'][dataset['safety'] == 'high']
print(mobil_bagus)

# # Menampilkan data dengan 3 filter
mobil_bagus = dataset[dataset['doors'] ==
                      '4'][dataset['persons'] == 'more'][dataset['safety'] == 'high']
print(mobil_bagus)
