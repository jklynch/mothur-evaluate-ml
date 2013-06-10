"""
A few functions for loading shared and design files.
"""
import numpy as np

class shared_data_group:
    def __init__(self):
        self.label_names = []
        self.group_names = []
        self.otu_column_names = None
        self.otu_frequency = None


def load_shared_file(shared_data_file_path):
    shared_data = shared_data_group()
    # read the shared file once to get the column names, label names, and group names
    with open(shared_data_file_path) as shared_data_file:
        shared_data.column_names = shared_data_file.readline().strip().split()
        for line in shared_data_file:
            label, group = line.strip().split(None, 2)[0:2]
            shared_data.label_names.append(label)
            shared_data.group_names.append(group)
    
    column_count = len(shared_data.column_names)
    print('read {} columns'.format(column_count))

    # read the shared file again to get OTU frequency data
    # skip the first row and the first three columns
    shared_data.otu_frequency = np.genfromtxt(
        shared_data_file_path,
        skip_header=1,
        usecols=xrange(3, column_count)
    )
    shared_data.otu_column_names = shared_data.column_names[3:]
    print('shared data shape: {}'.format(shared_data.otu_frequency.shape))
    print('otu column names length: {}'.format(len(shared_data.otu_column_names)))

    # python lists: label_names, group_names, otu_column_names
    # shared_data is a numpy array
    return shared_data


class design_data_group:
    def __init__(self):
        self.group_names = []
        self.treatment_names = []
        self.class_names = None
        self.class_name_to_number = None
        self.class_number_for_row = None


def load_design_file(design_file_path):
    design_data = design_data_group()

    with open(design_file_path) as design_file:
        for line in design_file:
            group, partition = line.strip().split()
            design_data.group_names.append(group)
            design_data.treatment_names.append(partition)

    # how many classes are there in the design file?
    # map each class name to a (floating point) number
    # we will want the class numbers in a NumPy array
    design_data.class_names = {x for x in design_data.treatment_names}
    print("class names: {}".format(design_data.class_names))
    design_data.class_name_to_number = {name:float(n) for n,name in enumerate(design_data.class_names)}
    print("class names and numbers: {}".format(design_data.class_name_to_number))
    # specify ndmin=2 and transpose() to get a column vector
    design_data.class_number_for_row = np.array(
        [design_data.class_name_to_number[class_name] for class_name in design_data.treatment_names],
        ndmin=2
    ).transpose()
    #print("class number for row:")
    #print(design_data.class_number_for_row[:,0])
    #print(treatment_names[0:5])
    return design_data


"""
Read a shared file and design file.  Write a single file
with OTU frequencies and treatments to be read with Octave.
For example, a shared file such as

  label	Group	         numOtus  Otu0001  Otu0002  Otu0003  Otu0004
  0.03	m6554_d0_Before	 2637	  162	   33	    15	     99
  0.03	m6554_d18_After1 2637	  128	   274	    188	     13
  0.03	m6554_d21_After1 2637	  40	   75	    215      29

and the corresponding design file

  m6554_d0_Before	Before
  m6554_d18_After1	After1
  m6554_d21_After1	After1

will be written as

  162  33  15  99  0
  128 274 188  13  1
   40  75 215  29  1

where in the last column 0 corresponds to 'Before' and 1 corresponds
to 'After1'.
"""
def write_combined_file(shared_file_path, design_file_path, output_file_path):
    shared_data = load_shared_file(shared_file_path)
    design_data = load_design_file(design_file_path)
    combined_data = np.hstack((shared_data.otu_frequency, design_data.class_number_for_row))
    np.savetxt(output_file_path, combined_data, fmt="%8d")


if __name__ == "__main__":
    import sys
    s = sys.argv[1]
    d = sys.argv[2]
    o = sys.argv[3]
    write_combined_file(s, d, o)
