"""
A few functions for loading shared and design files.
"""
import numpy as np


def load_shared_file(shared_data_file_path):
    with open(shared_data_file_path) as shared_data_file:
        column_names = shared_data_file.readline().strip().split()
        label_names = []
        group_names = []
        for line in shared_data_file:
            label, group = line.strip().split(None, 2)[0:2]
            label_names.append(label)
            group_names.append(group)
    
    column_count = len(column_names);
    print('read {} columns'.format(column_count))

    # read the OTU frequency data
    shared_data = np.genfromtxt(
        shared_data_file_path,
        skip_header=1,
        usecols=xrange(3, column_count)
    )
    otu_column_names = column_names[3:]
    print('shared data shape: {}'.format(shared_data.shape))
    print('otu column names length: {}'.format(len(otu_column_names)))

    print(column_names[0:5])
    print(label_names[0:5])
    print(group_names[0:5])
    
    print(shared_data[0:5, 0:5])

    return label_names, group_names, otu_column_names, shared_data


def load_design_file(design_file_path):
    with open(design_file_path) as design_file:
        group_names = []
        partition_names = []
        for line in design_file:
            group, partition = line.strip().split()
            group_names.append(group)
            partition_names.append(partition)

    print(group_names[0:5])
    print(partition_names[0:5])
    return group_names, partition_names