import pylab as pl
import numpy as np
import sklearn.preprocessing

import mothur_files
import smo


def svm_hmp_2_feature_plot():
    print('hazzah!')

    shared_file_path = '/home/jlynch/gsoc2013/data/Stool.0.03.subsample.0.03.filter.shared';
    design_file_path = '/home/jlynch/gsoc2013/data/Stool.0.03.subsample.0.03.filter.mix.design';

    shared_data = mothur_files.load_shared_file(shared_file_path)
    design_data = mothur_files.load_design_file(design_file_path)

    otu1 = 'Otu29878'
    otu2 = 'Otu29552'

    # where are Otu29741 and Otu29678
    n_otu1 = shared_data.otu_column_names.index(otu1)
    n_otu2 = shared_data.otu_column_names.index(otu2)

    print('{} is on column {}'.format(otu1, n_otu1))
    print('{} is on column {}'.format(otu2, n_otu2))

    print('shape of design_data.class_number_for_row {}'.format(design_data.class_number_for_row.shape))
    class_zero = design_data.class_number_for_row == 2.0
    class_one =  design_data.class_number_for_row == 1.0
    print('class zero count: {}'.format(np.sum(class_zero)))
    print('class one count: {}'.format(np.sum(class_one)))
    two_labels = np.logical_or(class_zero, class_one)
    print('shape of two_labels: {}'.format(two_labels.shape));
    label_index = np.arange(design_data.class_number_for_row.shape[0])
    reduced_label_index = label_index[two_labels[:,0]]
    print('reduced_label_index: {}'.format(reduced_label_index))

    two_labels_otu_frequency = shared_data.otu_frequency[reduced_label_index,:]
    print('shape of two_labels_otu_frequency: {}'.format(two_labels_otu_frequency.shape))

    reduced_otu_frequency = two_labels_otu_frequency[:,[n_otu1, n_otu2]]
    print('shaped of reduced_otu_frequency: {}'.format(reduced_otu_frequency.shape))
    #print('reduced_otu_frequency:\n{}'.format(reduced_otu_frequency))
    scaler = sklearn.preprocessing.StandardScaler()
    # the scaler returns a copy by default
    #X = scaler.fit_transform(reduced_otu_frequency)

    #exit()

    # the next line is pretty good
    # smo.smo(reduced_otu_frequency, design_data.class_number_for_row[two_labels], 0.5)
    smo.smo(reduced_otu_frequency, design_data.class_number_for_row[two_labels], 0.5)

    pl.xlabel(otu1)
    pl.ylabel(otu2)
    pl.gca().set_xticklabels([])
    pl.gca().set_yticklabels([])
    pl.show()



if __name__ == '__main__':
    svm_hmp_2_feature_plot()