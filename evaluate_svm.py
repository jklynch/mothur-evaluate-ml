"""
usage: python evaluate_svm.py shared-file-path design-file-path
"""

import argparse

import numpy as np

import matplotlib.pylab as pylab

import sklearn.svm
import sklearn.preprocessing
import sklearn.grid_search

import sklearn.cross_validation
import sklearn.metrics

import mothur_files


def evaluate_svm():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("shared_file_path", help="<path to shared file>")
    argparser.add_argument("design_file_path", help="<path to design file>")
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

    C_range = 10.0 ** np.arange(-3, 3)
    gamma_range = 10.0 ** np.arange(-5, -3)
    degree_range = np.arange(1, 5)
    coef0_range = np.arange(-3.0, 3.0)

    support_vector_machine(X, y, y_labels, "linear", dict(C=C_range))
    support_vector_machine(X, y, y_labels, "rbf", dict(gamma=gamma_range, C=C_range))
    support_vector_machine(X, y, y_labels, "poly", dict(C=C_range, degree=degree_range, coef0=coef0_range))
    #support_vector_machine(X, y, design_data, "sigmoid", dict(C=C_range, coef0=coef0_range))

    #rfe(X, design_data.class_number_for_row[:,0])
    #evaluate_linear_svm(X, design_data.class_number_for_row)


"""
This function fits a SVM model but no feature selection is done here.  This
is really just to determine the classification performance.
"""
def support_vector_machine(X, y, y_labels, kernel, param_grid):
    sss = sklearn.cross_validation.StratifiedShuffleSplit(
        y, test_size=0.5
    )
    train_index, test_index = next(iter(sss))
    X_train = X[train_index, :]
    X_test = X[test_index, :]
    y_train = y[train_index]
    y_test = y[test_index]

    #cv = sklearn.cross_validation.StratifiedKFold(y=y, n_folds=10)
    clf = sklearn.grid_search.GridSearchCV(
        sklearn.svm.SVC(kernel=kernel),
        param_grid=param_grid,
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

    print("Detailed classification report for kernel {}:".format(kernel))
    print('')
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print('')
    y_true, y_pred = y_test, clf.predict(X_test)
    print(sklearn.metrics.classification_report(y_true, y_pred, target_names=y_labels))
    print('')
    #print("The best {} SVM classifier is: {}".format(kernel, grid.best_estimator_))
    #print('best classifier score: {}'.format(grid.best_score_))

    #classifier = grid.best_estimator_
    #print("support_vectors_.shape: {}".format(classifier.support_vectors_.shape))
    #print("support_.shape: {}".format(classifier.support_.shape))
    #print("n_support_: {}".format(classifier.n_support_))
    #print("dual_coef_.shape: {}".format(classifier.dual_coef_.shape))
    #print("coef_.shape: {}".format(classifier.coef_.shape))


def evaluate_linear_svm(X, y):
    print("y.shape {}".format(y.shape))
    # use 10-fold cross validation
    k = 5
    ## repeat 100 times?
    ##N = 100
    # use random permutations of indices to select training and test sets
    # observation_indices will have the same shape as y
    observation_indices = np.array(np.arange(y.shape[0]))
    permuted_observation_indices = np.random.permutation(y.shape[0])
    print("observation_indices.shape {}".format(observation_indices.shape))
    test_set_size = int(y.shape[0] / k)
    print("test set size {}".format(test_set_size))
    # make and array of fold indices, eg 3-fold array for 12 elements:
    #   [1 2 3 1 2 3 1 2 3 1 2 3]
    k_fold_indices = np.mod(observation_indices, k)
    print("k_fold_indices {} {}".format(k_fold_indices.shape, k_fold_indices))
    # here is the list of Cs we will try
    C_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    score_for_C = np.zeros((k*len(C_list), 2))
    n = -1
    for C in C_list:
        for fold in np.arange(k):
            training_indices = permuted_observation_indices[ observation_indices != fold ]
            testing_indices = permuted_observation_indices[ observation_indices == fold ]
            svc = sklearn.svm.SVC(C=C, kernel='linear')
            svc.fit(X[training_indices, :], y[training_indices])
            score = svc.score(X[testing_indices, :], y[testing_indices])
            print('C:{} linear svm score: {}'.format(C, score))
            n += 1
            score_for_C[n, 0] = C
            score_for_C[n, 1] = score

    # plot results
    pylab.plot(score_for_C[:,0], score_for_C[:,1])
    pylab.show()


def rfe(X, y):
    cv = sklearn.cross_validation.StratifiedKFold(y=y, n_folds=10)
    rfesvm = sklearn.svm.SVC(
        kernel='rbf',
        C=100.0,
        gamma=1e-5,
    )
    rfesvm.fit(X, y)

    print("SVM classifier: {}".format(rfesvm))
    print('classifier score: {}'.format(rfesvm.score_))

    print("support_vectors_.shape: {}".format(rfesvm.support_vectors_.shape))
    print("support_.shape: {}".format(rfesvm.support_.shape))
    print("n_support_: {}".format(rfesvm.n_support_))
    print("dual_coef_.shape: {}".format(rfesvm.dual_coef_.shape))
    print("coef_.shape: {}".format(rfesvm.coef_.shape))


if __name__ == '__main__':
    evaluate_svm()
