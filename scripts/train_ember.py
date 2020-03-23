#!/usr/bin/env python

import os
import json
import ember
import pickle
import argparse
import matplotlib.pyplot as plt
import lightgbm as lgb


def main():
    prog = "train_ember"
    descr = "Train an ember model from a directory with raw feature files"
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("--modelname", type=str, default="SGD", help="Model name")
    parser.add_argument("-v", "--featureversion", type=int, default=2, help="EMBER feature version")
    parser.add_argument("datadir", metavar="DATADIR", type=str, help="Directory with raw features")
    parser.add_argument("--optimize", help="gridsearch to find best parameters", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.datadir) or not os.path.isdir(args.datadir):
        parser.error("{} is not a directory with raw feature files".format(args.datadir))

    X_train_path = os.path.join(args.datadir, f"X_train_{args.featureversion}.dat")
    y_train_path = os.path.join(args.datadir, f"y_train_{args.featureversion}.dat")
    # if they don't exist, compute them.
    if not (os.path.exists(X_train_path) and os.path.exists(y_train_path)):
        print("Creating vectorized features")
        ember.create_vectorized_features(args.datadir, args.featureversion)

    #feature_name = ['feature_' + str(col) for col in range(num_feature)]

    params = {
        "boosting": "gbdt",
        "objective": "regression",
        "num_iterations": 1000,
        "learning_rate": 0.05,
        "num_leaves": 2048,
        "max_depth": 15,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.5,
        "num_threads": 2,
    }
    if args.optimize:
        params = ember.optimize_model(args.datadir)
        print("Best parameters: ")
        print(json.dumps(params, indent=2))

    print("Training Classifier model")   
    lgbm_model = ember.train_model(args.datadir, params, args.featureversion)

    # Save to file in the current working directory
    #pkl_filename = os.path.join(args.datadir,f"{args.modelname}_model_{args.featureversion}.pkl")
   # with open(pkl_filename, 'wb') as f:
    #    pickle.dump(lgbm_model, f)
    print(f"file dumped into model.txt .... ")
    lgbm_model.save_model(os.path.join(args.datadir, f"model_{args.featureversion}.txt"))

    print('Plotting feature importances...')
    ax = lgb.plot_importance(lgbm_model, max_num_features=10)
    plt.savefig(f'lgbm_importances-0{args.featureversion}.png')

    # run
    os.system(f"xdg-open lgbm_importances-0{args.featureversion}.png")

if __name__ == "__main__":
    main()
