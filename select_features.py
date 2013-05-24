"""
usage: python select_features.py

No command line arguments.  This program expects to find two data files in a
subdirectory called 'data'.
"""

import numpy as np

import sklearn.svm
import sklearn.preprocessing
import sklearn.grid_search

import sklearn.cross_validation
import sklearn.linear_model

import mothur_files


def select_features():
    shared_data_file_path = 'data/Stool.0.03.subsample.0.03.filter.shared'
    design_data_file_path = 'data/Stool.0.03.subsample.0.03.filter.mix.design'
    
    shared_label_names, shared_group_names, otu_column_names, shared_data = \
        mothur_files.load_shared_file(shared_data_file_path)
    
    design_group_names, design_partition_names = \
        mothur_files.load_design_file(design_data_file_path)

    elastic_net(shared_data, otu_column_names, design_partition_names)
    linear_support_vector_machine(shared_data, otu_column_names, design_partition_names)
    rbf_support_vector_machine(shared_data, otu_column_names, design_partition_names)

"""
This function performs feature selection using elastic net.  The hyperparameters
l1_ratio and alpha are determined by cross validation.
"""
def elastic_net(shared_data, otu_column_names, design_partition_names):
    # scale the data to have mean 0 and std 1 in each feature    
    scaler = sklearn.preprocessing.StandardScaler()
    X = scaler.fit_transform(shared_data)
    # convert the last character of each partition name to a float - not elegant
    y = np.array(map(lambda x: float(x[-1]), design_partition_names))
    
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
        model_score = model.score(X[:n_train,:], y[:n_train])
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

"""
This function fits a SVM model but no feature selection is done here.  This
is really just to determine the classification performance.
"""
def rbf_support_vector_machine(shared_data, otu_column_names, design_partition_names):
    # scale the data to have mean 0 and std 1 in each feature
    scaler = sklearn.preprocessing.StandardScaler()
    X = scaler.fit_transform(shared_data)
    # convert the last character of each partition name to a float - not elegant
    y = np.array(map(lambda x: float(x[-1]), design_partition_names))

    C_range = 10.0 ** np.arange(-1, 2)
    gamma_range = 10.0 ** np.arange(-5, -3)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = sklearn.cross_validation.StratifiedKFold(y=y, n_folds=10)
    grid = sklearn.grid_search.GridSearchCV(
        sklearn.svm.SVC(kernel='rbf'),
        param_grid=param_grid,
        cv=cv,
        verbose=False
    )
    grid.fit(X, y)
    
    print("The best RBF SVM classifier is: ", grid.best_estimator_)
    print('best classifier score: {}'.format(grid.best_score_))


"""
I tried the following code using the scikit-learn RFE but it takes a long
time and the results were not consistent with classify.shared and elastic
net

    rfecv = sklearn.feature_selection.RFECV(
        estimator=sklearn.svm.SVC(C=0.01, kernel='linear'),
        cv=3,
        verbose=True
    )
    rfecv.fit(X, y)
    sorted_ranking_ndx = np.argsort(rfecv.ranking_)
    for n,i in enumerate(sorted_ranking_ndx[:20]):
         print('{} {} {}'.format(n, otu_column_names[i], rfecv.ranking_[i]))



This function performs feature selection using SVM-RFE.  I found a good C
parameter by cross-validation and then hard-coded it.
"""
def linear_support_vector_machine(shared_data, otu_column_names, design_partition_names):
    scaler = sklearn.preprocessing.StandardScaler()
    # the scaler returns a copy by default    
    X = scaler.fit_transform(shared_data)
    # convert the last character of each partition name to a float - not elegant
    y = np.array(map(lambda x: float(x[-1]), design_partition_names))

    remaining_otu_list = np.arange(len(otu_column_names))
    n_train = 400

    removed_feature_list = []
    while len(remaining_otu_list) > 0:
        svc = sklearn.svm.SVC(C=0.01, kernel='linear')    
        svc.fit(X[:n_train, remaining_otu_list], y[:n_train])
        #print('linear svm score: {}'.format(svc.score(X[n_train:, remaining_otu_list], y[n_train:])))
        
        #w_squared = svc.coef_.sum(axis=0)**2
        w_squared = (svc.coef_**2).sum(axis=0)
        w_squared_min_ndx = np.argmin(w_squared)
        otu_to_remove_ndx = remaining_otu_list[w_squared_min_ndx]
        otu_to_remove = otu_column_names[otu_to_remove_ndx]
        #print('removing {}'.format(otu_to_remove))
        remaining_otu_list = np.delete(remaining_otu_list, w_squared_min_ndx)
        removed_feature_list.append(otu_to_remove)
    
    removed_feature_list.reverse()
    
    # calculate a rank value by removing each feature
    svc = sklearn.svm.SVC(C=0.01, kernel='linear')
    svc.fit(X[:n_train, :], y[:n_train])
    all_features_score = svc.score(X[n_train:, :], y[n_train:])
    print('linear SVM score {}'.format(all_features_score))
    print('features ranked by linear SVM-RFE:')
    print(' n OTU')
    for n, otu_name in enumerate(removed_feature_list[:50]):
        print('{:2d} {}'.format(n, otu_name))
        #svc = sklearn.svm.SVC(C=0.01, kernel='linear')
        #otu_ndx = otu_column_names.index(otu_name)
        #print('otu_ndx for {}: {}'.format(otu_name, otu_ndx))
        #reduced_otu_list = range(len(otu_column_names))
        #reduced_otu_list.remove(otu_ndx)
        #svc.fit(X[:n_train, np.array([1, otu_ndx])], y[:n_train])
        #score = svc.score(X[n_train:, np.array([1, otu_ndx])], y[n_train:])
        #print('{:2d} {} {:4.2f}'.format(n, otu_name, all_features_score/score))



if __name__ == '__main__':
    select_features()