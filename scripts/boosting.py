import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    df_heart = pd.read_csv('data/heart.csv')

    # print(df_heart.target.describe())

    X, y = df_heart.drop('target', axis=1), df_heart.target
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.35)


    boost = GradientBoostingClassifier(n_estimators=50)   ## Este metodo uitliza arboles de decision simples
                                                          ## Su diferencia respecto al RandomForest es que
                                                          ## este aprende de los errores, no es de votacion, es secuencial
    
    boost.fit(X_train, y_train)
    boost_pred = boost.predict(X_test)

    print(accuracy_score(y_test, boost_pred))

    ## NOTEMOS LA EXAGERADA MEJORIA

    '''
    Resultado por consola44
    0.9164345403899722
    '''