from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(
    "C:/Users/overm/Documents/Kuliah/Semester 4/Data Mining/Pertemuan 5/diabetes.csv")      # memanggil data
x = dataset.iloc[:, [1, 7]].values             # menentukan data yang digunakan
# menentukan data yang dijadikan label
y = dataset.iloc[:, -1].values

# print(x)

x_train, x_test, y_train, y_test = train_test_split(            # memisahkan data test dengan data train
    x, y, test_size=0.2, random_state=0)

# print(x_train)
# print(x_test)
# print(y_test)

sc = StandardScaler()       # buat standatd scaler
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# print(x_test)

classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# menghitung nilai akurasi (nilai 1,1 ditambah 2,2 dibagi nilai semua)
cm = confusion_matrix(y_test, y_pred)

# print(cm)

# membuat diagram untuk data training
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(
    x1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Klasifikasi Data untuk Data Training')
plt.xlabel('Glukosa')
plt.ylabel('Umur')
plt.legend()
plt.show()

# membuat diagram ubtuk data test
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(
    x1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Klasifikasi Data untuk Data Testing')
plt.xlabel('Glukosa')
plt.ylabel('Umur')
plt.legend()
plt.show()
