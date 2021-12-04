import argparse

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump
def train_gradient_boost(model_name):
    '''
    :return: Void
    '''
    trial_x = np.loadtxt('trial_features.txt', dtype=int)
    trial_y = np.loadtxt('trial_features_target.txt', dtype=float)
    train_x = np.loadtxt('train_features.txt', dtype=int)
    train_y = np.loadtxt('train_features_target.txt', dtype=float)
    gbr = GradientBoostingRegressor(random_state=1337)
    print(len(np.append(trial_x,train_x, axis = 0)))
    gbr.fit(np.append(trial_x,train_x, axis = 0), np.append(trial_y,train_y))
    dump(gbr, model_name+".joblib")
def train_random_forest(model_name):
    '''
    :return: Void
    '''
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SEMEVAL2021 Task 1, models trained with features")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["GBR", "RF"],
        help="Select model to use",
        required=True,
    )
    parser.add_argument(
        "-mn",
        "--model_name",
        type=str,
        help="Specify name to save the model trained",
        required=True,
    )
    args = parser.parse_args()
    print(args)
    if args.model == "GBR":
        train_gradient_boost(args.model_name)
    elif args.model == "RF":
        train_random_forest(args.model_name)