"""
usage: python evaluate_enet.py shared-file-path design-file-path
"""

import argparse
import collections
import itertools

import numpy as np

import matplotlib.pylab as pylab

import sklearn.linear_model
import sklearn.preprocessing
import sklearn.grid_search
import sklearn.metrics

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

    #scaler = sklearn.preprocessing.StandardScaler()
    # the scaler returns a copy by default
    #X = scaler.fit_transform(shared_data.otu_frequency)
    #y = design_data.class_number_for_row[:,0]
    #y_labels = [design_data.class_number_to_name[n] for n in sorted(design_data.class_number_to_name.keys())]
    #print(y_labels)
    elastic_net_ovo(shared_data, design_data)
    #elastic_net(X, y, y_labels, shared_data.otu_column_names)


class two_class_elastic_net:
    def __init__(self, class_a, class_b, model):
        self.class_min = min(class_a, class_b)
        self.class_max = max(class_a, class_b)
        self.discriminant = (self.class_min + self.class_max) / 2.0
        self.model = model

    def classify(self, X):
        p = self.model.predict(X)
        # create an array to hold the classifications
        c = np.nan * np.ones(X.shape[0])
        c[p <  self.discriminant] = self.class_min
        c[p >= self.discriminant] = self.class_max
        if np.any(np.isnan(c)):
            raise Exception('failed to classify all observations')
        return c


def elastic_net_ovo(shared_data, design_data):
    scaler = sklearn.preprocessing.StandardScaler()
    # the scaler returns a copy by default
    X = scaler.fit_transform(shared_data.otu_frequency)
    y = design_data.class_number_for_row[:,0]

    # split the data:
    #   2/3 for training the one-versus-one models
    #   1/3 for testing the composite model
    sss = sklearn.cross_validation.StratifiedShuffleSplit(
        y, test_size=1.0/3.0
    )
    train_index, test_index = next(iter(sss))
    X_train = X[train_index, :]
    X_test = X[test_index, :]
    y_train = y[train_index]
    y_test = y[test_index]

    elastic_net_models = []
    # the names of the classes are in the set design_data.class_names
    for class_name_pair in itertools.combinations(design_data.class_names, 2):
        print('class name pair: {}'.format(class_name_pair))
        # fit an elastic net model to this class pair
        class_a_name = class_name_pair[0]
        class_a_number = design_data.class_name_to_number[class_a_name]
        class_a_mask = y_train == class_a_number
        print('found {} observations for class {}'.format(sum(class_a_mask), class_a_name))
        class_b_name = class_name_pair[1]
        class_b_number = design_data.class_name_to_number[class_b_name]
        class_b_mask = y_train == class_b_number
        print('found {} observations for class {}'.format(sum(class_b_mask), class_b_name))
        class_pair_mask = np.logical_or(class_a_mask, class_b_mask)
        X_train_class_pair = X_train[class_pair_mask, :]
        y_train_class_pair = y_train[class_pair_mask]
        elastic_net_models.append(
            elastic_net_2_class(
                X_train_class_pair,
                y_train_class_pair,
                shared_data,
                design_data
            )
        )

    # test the composite model
    test_classification_votes = np.nan * np.ones((X_test.shape[0], len(elastic_net_models)))
    for i, elastic_net_model in enumerate(elastic_net_models):
        test_classification_votes[:, i] = elastic_net_model.classify(X_test)

    test_classification = np.nan * np.ones(X_test.shape[0])
    for i in range(test_classification_votes.shape[0]):
        vote_count = collections.Counter()
        for vote in test_classification_votes[i, :]:
            vote_count[vote] += 1
        top_vote = vote_count.most_common(1)
        #print('votes: {}'.format(test_classification_votes[i, :]))
        #print('  top vote: {}'.format(top_vote))
        if len(top_vote) == 1:
            test_classification[i] = top_vote[0][0]
        else:
            pass

    print(
        sklearn.metrics.classification_report(
            y_test,
            test_classification
        )
    ) #, target_names=y_labels))


"""
This function performs feature selection using elastic net.  The hyperparameters
l1_ratio and alpha are determined by cross validation.
"""
def elastic_net_2_class(X, y, shared_data, design_data):
    sss = sklearn.cross_validation.StratifiedShuffleSplit(
        y, test_size=0.5
    )
    train_index, test_index = next(iter(sss))
    X_train = X[train_index, :]
    X_test = X[test_index, :]
    y_train = y[train_index]
    y_test = y[test_index]

    # we expect there are exactly 2 classes
    # if the y array in fact has exactly 2 classes then two_class_set will look something like this:
    #   [(1.0, 2.0)]
    two_class_list = list(set(y))
    class_count = len(two_class_list)
    if class_count == 2:
        print('found two classes: {}'.format(two_class_list))
    else:
        raise Exception('expected 2 classes but found {}'.format(two_class_list))

    best_two_class_model = None
    best_model_score = 0.0
    # I could not get ElasticNetCV to choose the best l1_ratio from a list
    # so I am doing that explictly with a loop
    for l1_ratio in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
        model = sklearn.linear_model.ElasticNetCV(
            l1_ratio=l1_ratio,
            cv=5, # having trouble with 10
            verbose=False
        ).fit(X_train, y_train)
        two_class_model = two_class_elastic_net(two_class_list[0], two_class_list[1], model)
        X_test_classification = two_class_model.classify(X_test)
        #test_prediction = model.predict(X_test)
        #test_classification = test_prediction.round()
        test_classification_correct = (y_test == X_test_classification)
        model_score = test_classification_correct.sum() / float(len(test_classification_correct))
        if model_score > best_model_score:
            best_two_class_model = two_class_model
            best_model_score = model_score

        #print('l1_ratio: {}'.format(model.l1_ratio_))
        #print('alpha:    {}'.format(model.alpha_))
        #print('model score: {}'.format(model_score))
    print('class discriminant: {}'.format(best_two_class_model.discriminant))
    print('feature ranking by elastic net:')
    print('   OTU      Rank')
    enet_top_feature_list = get_enet_top_features(best_two_class_model.model)
    for n, (feature_ndx, rank) in enumerate(enet_top_feature_list[:10]):
        print('{:2d} {} {:4.2f}'.format(n, shared_data.otu_column_names[feature_ndx], rank))

    y_true, y_pred = y_test, best_two_class_model.classify(X_test)
    # it helps to make the labels integers ???
    print(sklearn.metrics.classification_report([int(y) for y in y_true], [int(y) for y in y_pred])) #, target_names=y_labels))
    print('')

    print('best l1_ratio: {}'.format(best_two_class_model.model.l1_ratio_))
    print('best alpha:    {}'.format(best_two_class_model.model.alpha_))

    return best_two_class_model


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

    best_model = None
    best_model_score = 0.0
    # I could not get ElasticNetCV to choose the best l1_ratio from a list
    # so I am doing that explictly with a loop
    for l1_ratio in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
        model = sklearn.linear_model.ElasticNetCV(
            l1_ratio=l1_ratio,
            #cv = sklearn.cross_validation.LeaveOneOut(len(train_index)),
            cv=5, # having trouble with 10
            verbose=False
        ).fit(X_train, y_train)
        # I think using model.score was the wrong way to go
        model_score_xx = model.score(X_test, y_test)
        #print('built-in model.score: {}'.format(model_score_xx))
        test_prediction = model.predict(X_test)
        test_classification = test_prediction.round()
        test_classification_correct = (y_test == test_classification)
        model_score = test_classification_correct.sum() / float(len(test_classification_correct))
        if model_score > best_model_score:
            best_model = model
            best_model_score = model_score

        #print('l1_ratio: {}'.format(model.l1_ratio_))
        #print('alpha:    {}'.format(model.alpha_))
        #print('model score: {}'.format(model_score))
    print('feature ranking by elastic net:')
    print('   OTU      Rank')
    enet_top_feature_list = get_enet_top_features(best_model)
    for n, (feature_ndx, rank) in enumerate(enet_top_feature_list[:10]):
        print('{:2d} {} {:4.2f}'.format(n, otu_column_names[feature_ndx], rank))

    y_true, y_pred = y_test, best_model.predict(X_test)
    # it helps to make the labels integers
    print(sklearn.metrics.classification_report([int(y) for y in y_true], [int(y) for y in y_pred], target_names=y_labels))
    print('')

    print('best l1_ratio: {}'.format(best_model.l1_ratio_))
    print('best alpha:    {}'.format(best_model.alpha_))


def get_enet_top_features(enet_model):
    sorted_coef_ndx = list(np.argsort(np.abs(enet_model.coef_), axis=0))
    sorted_coef_ndx.reverse()
    abs_coef_sum = np.sum(np.abs(enet_model.coef_))
    return [(i, 100.0*np.abs(enet_model.coef_[i])/abs_coef_sum) for i in sorted_coef_ndx]    


if __name__ == '__main__':
    evaluate_enet()
