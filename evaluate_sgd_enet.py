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


def evaluate_sgd_enet():
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
    sgd_elastic_net(X, y, y_labels, shared_data.otu_column_names)


"""
This function performs feature selection using elastic net.  The hyperparameters
l1_ratio and alpha are determined by cross validation.
"""
def sgd_elastic_net(X, y, y_labels, otu_column_names):
    sss = sklearn.cross_validation.StratifiedShuffleSplit(
        y, test_size=0.5
    )
    train_index, test_index = next(iter(sss))
    X_train = X[train_index, :]
    X_test = X[test_index, :]
    y_train = y[train_index]
    y_test = y[test_index]

    #cv = sklearn.cross_validation.StratifiedKFold(y=y, n_folds=10)
    alpha_range = 10.0**-np.arange(2,4) # 0.001 is what we get
    #warm_start_range = [True, False]
    n_iter = np.ceil(10**6 / y_train.shape[0])
    clf = sklearn.grid_search.GridSearchCV(
        sklearn.linear_model.SGDClassifier(penalty='elasticnet', n_iter=n_iter, shuffle=True, warm_start=True),
        #sklearn.svm.SVC(kernel=kernel),
        param_grid=dict(alpha=alpha_range),
        verbose=False
    )
    clf.fit(
        X_train,
        y_train,
        #cv = sklearn.cross_validation.LeaveOneOut(len(train_index))
        cv=10
    )

    print("Best parameters set found on development set:")
    print('')
    print(clf.best_estimator_)
    print('')
    #print("Grid scores on development set:")
    #print('')
    #for params, mean_score, scores in clf.grid_scores_:
    #    print("%0.3f (+/-%0.03f) for %r" % (
    #        mean_score, scores.std() / 2, params))
    #print('')

    print("Detailed classification report")
    print('')
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print('')
    y_true, y_pred = y_test, clf.predict(X_test)
    print(sklearn.metrics.classification_report(y_true, y_pred, target_names=y_labels))
    print('')
    print('coef_.shape: {}'.format(clf.best_estimator_.coef_.shape))

    print('feature ranking by elastic net:')
    print('   OTU      Rank')
    enet_top_feature_list = get_enet_top_features(clf.best_estimator_)
    for n, (feature_ndx, rank) in enumerate(enet_top_feature_list[:50]):
        print('{:2d} {} {:4.2f}'.format(n, otu_column_names[feature_ndx], rank))


def get_enet_top_features(enet_model):
    print(enet_model.coef_.shape)
    print(np.abs(enet_model.coef_).shape)
    print(np.sum(np.abs(enet_model.coef_), axis=0).shape)
    sum_abs_coef = np.sum(np.abs(enet_model.coef_), axis=0)
    sorted_coef_ndx = list(np.argsort(sum_abs_coef))
    sorted_coef_ndx.reverse()
    abs_coef_sum = np.sum(sum_abs_coef)
    return [(i, 100.0*np.abs(sum_abs_coef[i])/abs_coef_sum) for i in sorted_coef_ndx]


if __name__ == '__main__':
    evaluate_sgd_enet()
