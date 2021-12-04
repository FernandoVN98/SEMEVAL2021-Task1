import argparse

import numpy as np
from joblib import load


def predict(model_name):
    '''
    :param model_name:
    :return:
    '''
    clf = load(model_name+'.joblib')
    test_x = np.loadtxt('test_features.txt', dtype=int)
    return clf.predict(test_x)
def evaluate(predicted,objective):
    '''
    :param predicted:
    :param objective:
    :return:
    '''
    pearson_corr = np.corrcoef(predicted,objective)
    print("The pearson correlation obtained with the model is: "+str(pearson_corr))
    return
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SEMEVAL2021 Task 1, models trained with features")
    parser.add_argument(
        "-mn",
        "--model_name",
        type=str,
        help="Specify name to load the model",
        required=True,
    )
    args = parser.parse_args()
    evaluate(predict(args.model_name),np.loadtxt('test_features_target.txt', dtype=float))