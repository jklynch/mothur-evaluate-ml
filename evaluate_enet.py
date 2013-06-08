"""
usage: python evaluate_enet.py shared-file-path design-file-path
"""

import argparse

import numpy as np

import matplotlib.pylab as pylab

import sklearn.linear_model
import sklearn.preprocessing
import sklearn.grid_search

import sklearn.cross_validation

import mothur_files


def evaluate_enet():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("shared_file_path")
    argparser.add_argument("design_file_path")
    args = argparser.parse_args()

    print("shared file path: {0.shared_file_path}".format(args))
    print("design file path: {0.design_file_path}".format(args))

    label_names, shared_group_names, otu_column_names, shared_data = mothur_files.load_shared_file(args.shared_file_path)
    design_group_name_for_row, treatment_name_for_row = mothur_files.load_design_file(args.design_file_path)

    # how many classes are there in the design file?
    # map each class name to a (floating point) number
    # we will want the class numbers in a NumPy array
    class_names = {x for x in treatment_name_for_row}
    print("class names: {}".format(class_names))
    class_name_to_number = {name:float(n) for n,name in enumerate(class_names)}
    print("class names and numbers: {}".format(class_name_to_number))
    class_number_for_row = np.array(
        [class_name_to_number[class_name] for class_name in treatment_name_for_row]
    )

    scaler = sklearn.preprocessing.StandardScaler()
    # the scaler returns a copy by default
    X = scaler.fit_transform(shared_data)
    # X and class_number_for_row are ready

    C_range = 10.0 ** np.arange(-3, 3)
    gamma_range = 10.0 ** np.arange(-5, -3)
    degree_range = np.arange(1, 4)
    coef0_range = np.arange(-3.0, 3.0)

    elastic_net(X, class_number_for_row, otu_column_names)


"""
This function performs feature selection using elastic net.  The hyperparameters
l1_ratio and alpha are determined by cross validation.
"""
def elastic_net(X, y, otu_column_names):
   
    ##############################################################################
    n_train = 400
    best_model = None
    best_model_score = 0.0
    # I could not get ElasticNetCV to choose the best l1_ratio from a list
    # so I am doing that explictly with a loop
    for l1_ratio in [.1, .2, .3, .4]:
        model = sklearn.linear_model.ElasticNetCV(
            l1_ratio=l1_ratio,
            cv=sklearn.cross_validation.StratifiedKFold(y[n_train:], 10),
            verbose=False
        ).fit(X[n_train:, :], y[n_train:])
        # I think using model.score was the wrong way to go
        # model_score = model.score(X[:n_train,:], y[:n_train])
        test_prediction = model.predict(X[:n_train])
        test_classification = test_prediction.round()
        test_classification_correct = (y[:n_train] == test_classification)
        model_score = test_classification_correct.sum() / float(len(test_classification_correct))
        if model_score > best_model_score:
            best_model = model
            best_model_score = model_score

        print('l1_ratio: {}'.format(model.l1_ratio_))
        print('alpha:    {}'.format(model.alpha_))
        print('model score: {}'.format(model_score))
        print('feature ranking by elastic net:')
        print('   OTU      Rank')
        enet_top_feature_list = get_enet_top_features(model)
        for n, (feature_ndx, rank) in enumerate(enet_top_feature_list[:50]):
            print('{:2d} {} {:4.2f}'.format(n, otu_column_names[feature_ndx], rank))
    
    print('best l1_ratio: {}'.format(best_model.l1_ratio_))
    print('best alpha:    {}'.format(best_model.alpha_))
    print('best model score: {}'.format(best_model_score))
    print('')


def get_enet_top_features(enet_model):
    sorted_coef_ndx = list(np.argsort(np.abs(enet_model.coef_), axis=0))
    sorted_coef_ndx.reverse()
    abs_coef_sum = np.sum(np.abs(enet_model.coef_))
    return [(i, 100.0*np.abs(enet_model.coef_[i])/abs_coef_sum) for i in sorted_coef_ndx]    


if __name__ == '__main__':
    evaluate_enet()
