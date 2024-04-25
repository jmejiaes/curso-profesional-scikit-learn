import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA

from sklearn.linear_model  import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    df_heart = pd.read_csv('data/heart.csv')

    X, y  = df_heart.drop('target', axis=1), df_heart.target

    X = StandardScaler().fit_transform(X)    ### Se deberia en realidad entrenar con X, o con X_train??

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(X_train.shape)
    print(y_train.shape)

    pca = PCA(n_components=3)    ### Si no se le pasa ningun parametro el n componentes es igual al numero de columnas.
    pca.fit(X_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10)  ## El batch size es el numero de muestras que se van a tomar en cada iteracion.
    ipca.fit(X_train)

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    # plt.plot(range(len(pca.explained_variance_)), np.cumsum(pca.explained_variance_ratio_))  ## Propondria que se haga con cumsum
    # plt.show()

    logistic = LogisticRegression(solver='lbfgs')  ## Parametro que evita errores en el futuro

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    logistic.fit(X_train_pca, y_train)
    print('Score PCA : ', logistic.score(X_test_pca, y_test))


    X_train_ipca = ipca.transform(X_train)
    X_test_ipca = ipca.transform(X_test)

    logistic.fit(X_train_ipca, y_train)
    print('Score IPCA : ', logistic.score(X_test_ipca, y_test))


    ## Clase 12, Kernels y PCA

    kpca = KernelPCA(n_components=4 , kernel='poly') ## El numero de componentes es opcional, el kernel 'lineal' seria equivalente al pca normal
    kpca.fit(X_train)

    X_train_kpca = kpca.transform(X_train)
    X_test_kpca = kpca.transform(X_test)

    logistic_kpca = LogisticRegression(solver='lbfgs')
    logistic_kpca.fit(X_train_kpca, y_train)

    print('Score KPCA : ', logistic_kpca.score(X_test_kpca, y_test))




