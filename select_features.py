import time

import numpy as np
import pylab as pl

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


def elastic_net(shared_data, otu_column_names, design_partition_names):
    shared_data_std = shared_data.std(0)
    shared_data_max_std = np.max(shared_data_std)
    shared_data_max_std_column_name = otu_column_names[np.argmax(shared_data_std)]
    print('max std {} is in column {}'.format(shared_data_max_std, shared_data_max_std_column_name))
    shared_data_min_std = np.min(shared_data_std)
    shared_data_min_std_column_name = otu_column_names[np.argmin(shared_data_std)]
    print('min std {} is in column {}'.format(shared_data_min_std, shared_data_min_std_column_name))
    
    X = shared_data / shared_data.std(0)
    # convert the last character of each partition name to a float - not elegant
    y = np.array(map(lambda x: float(x[-1]), design_partition_names))
    
    ##############################################################################
    # ElasticNetCV
    
    # Compute paths
    print "Computing regularization path using coordinate descent elastic net..."
    t1 = time.time()
    model = sklearn.linear_model.ElasticNetCV(
        l1_ratio=[0.1, 1.0],
        #l1_ratio=[.1, .5, .7, .9, .95, .99, 1.0],
        #cv=10,
        cv=sklearn.cross_validation.StratifiedKFold(y, 10),    
        normalize=False,
        verbose=True
    ).fit(X, y)
    t_enet_cv = time.time() - t1
    
    print('best l1_ratio: {}'.format(model.l1_ratio_))
    print('best alpha:    {}'.format(model.alpha_))
    
    print('model score: {}'.format(model.score(X, y)))
    
    l1_ratio_ndx = model.l1_ratio.index(model.l1_ratio_)
    
    sorted_coef_ndx = np.argsort(np.abs(model.coef_), axis=0)
    
    print('max coef is in column {}'.format(otu_column_names[sorted_coef_ndx[:,0]]))
    
    # Display results
    m_log_alphas = -np.log10(model.alphas_)
    
    pl.figure()
    ymin, ymax = -10, 10
    print('model.mse_path_.shape: {}'.format(model.mse_path_.shape))
    pl.plot(m_log_alphas, model.mse_path_[l1_ratio_ndx], ':')
    pl.plot(m_log_alphas, model.mse_path_[l1_ratio_ndx].mean(axis=-1), 'k',
            label='Average across the folds', linewidth=2)
    pl.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
               label='alpha: CV estimate')
    
    pl.legend()
    
    pl.xlabel('-log(alpha)')
    pl.ylabel('Mean square error')
    pl.title('Mean square error on each fold: coordinate descent '
             '(train time: %.2fs)' % t_enet_cv)
    pl.axis('tight')
    pl.ylim(ymin, ymax)
    pl.show()

if __name__ == '__main__':
    select_features()