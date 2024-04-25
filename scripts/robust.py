import pandas as pd

from sklearn.linear_model import (
    RANSACRegressor, HuberRegressor
)

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('data/felicidad_corrupt.csv')
    print(dataset.head())

    X, y = dataset.drop('score', axis = 1), dataset.score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

       ## La manera mas proifesional de usar varios estimadores
    estimators = {  
        'SVR' : SVR(gamma='auto', C = 1.0, epsilon=0.1),
        'RANSAC' : RANSACRegressor(),   ## Recordemos que ransac es un metae stimador, por lo que se le pueden
                                        ## Pasar estimadores como el SVR como parametros, por defecto trabaja linear regression
        'HUBER' : HuberRegressor(epsilon=1.35)  ## Si este parametro es mas bajo, muchos menos datos seran considerados valores atipicos
    }

    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(y_test)
        mse = mean_squared_error(y_test, predictions)

        print('-'*32)
        print(f'MSE para {name}:', mse )
