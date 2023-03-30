# Mengubah data menjadi matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer       # Mengubah data menjadi matrix
from sklearn.impute import SimpleImputer
import numpy as np                  # Numpy digunakan untuk komputasi matriks
import matplotlib.pyplot as plt
# Matplotlib merupakan library python untuk presentasi data berupa grafik atau plot
import pandas as pd

dataset = pd.read_csv(
    'C:/Users/overm/Documents/Kuliah/Semester 4/Data Mining/Pertemuan 3/car2.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(x[:, 1:6])
# x[:, 1:6] = imputer.transform(x[:, 1:6])

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0, 1, 2, 3, 4])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# print(x)

le = LabelEncoder()
y = le.fit_transform(y)

# print(y)

# Untuk memisahkan data training dan data testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)

# print(y_train)

# Untuk scale ulang angka agar gapnya tidak terlalu besar
# sc = StandardScaler()
# x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
# x_test[:, 3:] = sc.fit_transform(x_test[:, 3:])
