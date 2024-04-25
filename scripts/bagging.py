import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    df_heart = pd.read_csv('data/heart.csv')

    # print(df_heart.target.describe())

    X, y = df_heart.drop('target', axis=1), df_heart.target
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.35)

    ## Notemos que este clasificador no es tan bueno por si solo, ya que se manda sin params
    knn_class = KNeighborsClassifier()
    knn_class.fit(X_train, y_train)
    
    knn_test_pred = knn_class.predict(X_test)

    print('-'*63)
    print(accuracy_score(y_test, knn_test_pred))

    ## Ahora, comparemos este resultado contra el de ensamble

    bag_class = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    bag_predict = bag_class.predict(X_test)

    print('-'*63)
    print(accuracy_score(y_test, bag_predict))

    ## Con los metodos de ensamble se puede llegar a muy buenos resultados por 
    ## Medio de la agregacion de modelos mas simples, incluso aunque no sean
    ## Los mejores modelos individualmente