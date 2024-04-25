import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score, KFold

if __name__ == '__main__':
    
    dataset = pd.read_csv('data/felicidad.csv')

    X, y = dataset.drop(['country', 'score'], axis = 1), dataset.score

    model = DecisionTreeRegressor()
    
    ## Antes se llamaba a la funcion .fit() sin embargo, como ya se utiliza Cross Validation

    score = cross_val_score(model, X, y, scoring='neg_mean_squared_error')
    print(score)

    '''
    [-0.41966018 -0.05615163 -0.07867811 -0.08574053 -0.43412454] -> por defecto está haciendo 5 folds, y este es el mse (negativo) para cada uno
    '''

    score = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
    print(np.abs(np.mean(score)))

    ## Ahora supongamos que queremos hacer el Kfold manual
                            ## Aleatoriamente shuffle
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for train, test in kf.split(dataset):   ## Asi se generan los pliegues de manera manual
        print(train)
        print(test)

        ## Si quisieramos pasarlas dentro de un modelo, se pasa de la misma manera, pero por cada split