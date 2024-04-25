import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == '__main__':

    dataset = pd.read_csv('data/candy.csv')
    
    X = dataset.drop('competitorname', axis=1)

    meanshift = MeanShift().fit(X)
    print(max(meanshift.labels_)+1)  ## Los labels devuelve un arreghlo de como categorizo, predijo, las cosas en el dataset
                                     ## Parece entonces que el fit_predct hace lo mismo solo que devuelve el labeles
                                    
    print('-'*64)
    print(meanshift.cluster_centers_)  ## Esto nos da un arreglo de k valores con el centro de cada uno de los features en cada cluster
    
    '''
    3
    ----------------------------------------------------------------
    [[2.25000000e-01 5.75000000e-01 1.00000000e-01 2.50000000e-02
    5.00000000e-02 2.50000000e-02 3.00000000e-01 1.00000000e-01
    5.50000000e-01 4.57599993e-01 3.67824996e-01 4.10442122e+01]
    [4.68750000e-01 5.00000000e-01 1.25000000e-01 1.56250000e-01
    9.37500000e-02 6.25000000e-02 1.25000000e-01 3.12500000e-01
    5.31250000e-01 4.57281243e-01 4.67874998e-01 5.21138597e+01]
    [8.26086957e-01 1.73913043e-01 3.04347826e-01 3.04347826e-01
    1.73913043e-01 1.73913043e-01 0.00000000e+00 5.21739130e-01
    4.34782609e-01 5.81391293e-01 6.38086963e-01 6.47120799e+01]]
    '''

    dataset['meanshift'] = meanshift.labels_
    print('-'*64)
    print(dataset)