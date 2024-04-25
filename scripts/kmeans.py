import pandas as pd

from sklearn.cluster import MiniBatchKMeans ## Da un resultado muy parecido a kmeans, solo que para pocos recursos computacionales

if __name__ == '__main__':

    dataset = pd.read_csv('data/candy.csv')

    X = dataset.drop('competitorname', axis=1) ## ESta variable no aporta info numeridcamente

    kmeans =  MiniBatchKMeans(n_clusters=4, batch_size=8)   ## imagine que es una dulceria que quiere obtener como organizar los dulces en 4 estanterias, de manera que queden parecidos
    kmeans.fit(X)

    print('total de centroides : ' , len(kmeans.cluster_centers_))
    print('-'*64)
    print(kmeans.predict(X))
    print('-'*64)
    
    dataset['resultant group'] = kmeans.predict(X)
    
    print(dataset)