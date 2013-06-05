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

import mothur_files


def evaluate_svm():
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

    support_vector_machine(X, class_number_for_row, "linear", dict(C=C_range))
    support_vector_machine(X, class_number_for_row, "rbf", dict(gamma=gamma_range, C=C_range))
    support_vector_machine(X, class_number_for_row, "poly", dict(C=C_range, degree=degree_range, coef0=coef0_range))
    support_vector_machine(X, class_number_for_row, "sigmoid", dict(C=C_range, coef0=coef0_range))

    #evaluate_linear_svm(X, class_number_for_row)


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


"""
This function fits a SVM model but no feature selection is done here.  This
is really just to determine the classification performance.
"""
def support_vector_machine(X, y, kernel, param_grid):
    n_train = 200
    cv = sklearn.cross_validation.StratifiedKFold(y=y, n_folds=10)
    grid = sklearn.grid_search.GridSearchCV(
        sklearn.svm.SVC(kernel=kernel),
        param_grid=param_grid,
        cv=cv,
        verbose=False
    )
    grid.fit(X, y)

    print("The best {} SVM classifier is: {}".format(kernel, grid.best_estimator_))
    print('best classifier score: {}'.format(grid.best_score_))


if __name__ == '__main__':
    evaluate_svm()