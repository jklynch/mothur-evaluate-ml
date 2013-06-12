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

    shared_data = mothur_files.load_shared_file(args.shared_file_path)
    design_data = mothur_files.load_design_file(args.design_file_path)

    scaler = sklearn.preprocessing.StandardScaler()
    # the scaler returns a copy by default
    X = scaler.fit_transform(shared_data.otu_frequency)
    y = design_data.class_number_for_row[:,0]
    y_labels = [design_data.class_number_to_name[n] for n in sorted(design_data.class_number_to_name.keys())]
    print(y_labels)
    elastic_net(X, y, y_labels, shared_data.otu_column_names)


"""
This function performs feature selection using elastic net.  The hyperparameters
l1_ratio and alpha are determined by cross validation.
"""
def elastic_net(X, y, y_labels, otu_column_names):
    sss = sklearn.cross_validation.StratifiedShuffleSplit(
        y, test_size=0.5
    )
    train_index, test_index = next(iter(sss))
    X_train = X[train_index, :]
    X_test = X[test_index, :]
    y_train = y[train_index]
    y_test = y[test_index]
   
    ##############################################################################
    #n_train = int( 4.0/5.0 * y.shape[0] )
    #print('n_train: {}'.format(n_train))
    best_model = None
    best_model_score = 0.0
    # I could not get ElasticNetCV to choose the best l1_ratio from a list
    # so I am doing that explictly with a loop
    for l1_ratio in [.1, .2, .3, .4]:
        model = sklearn.linear_model.ElasticNetCV(
            l1_ratio=l1_ratio,
            #cv=sklearn.cross_validation.StratifiedKFold(y[n_train:], 3),
            cv=5,
            verbose=False
        ).fit(X_train, y_train)
        # I think using model.score was the wrong way to go
        model_score_xx = model.score(X_test, y_test)
        print('built-in model.score: {}'.format(model_score_xx))
        test_prediction = model.predict(X_test)
        test_classification = test_prediction.round()
        test_classification_correct = (y_test == test_classification)
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
        for n, (feature_ndx, rank) in enumerate(enet_top_feature_list[:10]):
            print('{:2d} {} {:4.2f}'.format(n, otu_column_names[feature_ndx], rank))

    y_true, y_pred = y_test, best_model.predict(X_test)
    print(sklearn.metrics.classification_report(y_true, y_pred))
    print('')

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
