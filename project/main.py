from utils import Utils
from models import Models

if __name__ == '__main__' : 

    utils = Utils()
    models = Models()

    path = './in/felicidad.csv'

    data = utils.load_from_csv(path)
    X, y = utils.features_target(data, ['country', 'score', 'rank'], 'score')

    models.grid_training(X, y)

    print(data)

