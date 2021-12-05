import argparse

import numpy as np
from joblib import load


def predict(model_name, ending_features):
    '''
    :param model_name: name of the file where is saved the model trained
    :return: array of predictions for the test data made with the loaded model
    '''
    clf = load(model_name+'.joblib')
    test_x = np.loadtxt('test_features'+ending_features+'.txt', dtype=int)
    return clf.predict(test_x)
def evaluate(predicted,objective):
    '''
    :param predicted: array of values predicted by the model for the test data
    :param objective: array of true values for the test data
    :return: Void
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
    parser.add_argument(
        "-ending",
        "--ending_features",
        type=str,
        help="Specify the ending of the name of the features file",
        required=True,
    )
    args = parser.parse_args()
    evaluate(predict(args.model_name, args.ending_features),np.loadtxt('test_features'+args.ending_features+'_target.txt', dtype=float))