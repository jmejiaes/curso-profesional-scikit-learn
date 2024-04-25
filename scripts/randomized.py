import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':

    dataset = pd.read_csv('data/felicidad.csv')

    reg = RandomForestRegressor()

    X, y = dataset.drop(['country', 'rank', 'score'], axis=1), dataset.score

    param_grid = {
        'n_estimators' : range(4,16),
        'criterion' : ['squared_error', 'absolute_error'],
        'max_depth' : range(2,11)    ## Notemos que no se toman arboles muy profundos, esto es debido a que para los ensambles se prefieren 
                                     ## Combinaciones de modelos simples
    }

    rand_est = RandomizedSearchCV(reg , param_grid, n_iter=100 , cv = 3, scoring='neg_mean_absolute_error').fit(X, y)
    
    rand_est

    print(rand_est.best_estimator_)
    print(rand_est.best_params_)

    '''
    RandomForestRegressor(max_depth=7, n_estimators=5)
    {'n_estimators': 5, 'max_depth': 7, 'criterion': 'squared_error'}
    '''

    ## Ya teniendo estos mejores parametros, no tenemos que volver a implementar tood el proceso, pues rand_est es un metaestimador
    ## Por lo que se pueden ejecutar los metodos de los etimadores en el

    print(rand_est.predict(X.loc[[0]]))

    ''' [7.5250001] -> Notemos que el valor original de esta observacion era de 7.59''' 