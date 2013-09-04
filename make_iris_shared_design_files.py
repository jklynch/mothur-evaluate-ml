import sklearn.datasets as skdata


def make_files():
    print('hazzah!')
    iris = skdata.load_iris()
    print(iris.keys())
    print(iris['feature_names'])
    print(iris['target_names'])
    print(iris['target'])
    print(iris['data'])
    print(iris['DESCR'])

    # start with ten rows in shared and design files
    with open('iris.shared', 'w') as iris_shared_file, open('iris.design', 'w') as iris_design_file:
        iris_data = iris['data']
        iris_target = iris['target']
        iris_target_names = iris['target_names']
        iris_feature_names = iris['feature_names']
        iris_feature_count = len(iris_feature_names)

        shared_headers = ['label', 'Group', 'numOtus']
        shared_headers.extend(['Otu{:02}'.format(n+1) for n in range(iris_feature_count)])
        # make the column width 10 with {:10}
        shared_header_line = '\t'.join(['{}'.format(header) for header in shared_headers])
        write_and_print(shared_header_line, iris_shared_file)

        # no header for design file
        # design_headers = []

        for i in range(len(iris_data)):
            target = iris_target[i]
            #shared_label = '{}.0'.format(target)
            shared_label = '1.0'
            shared_group = '{}.{}'.format(iris_target_names[target],i)
            shared_num_otus = '{}'.format(iris_feature_count)
            otus = ['{}'.format(int(10*feature)) for feature in iris_data[i]]
            shared_line_columns = [shared_label, shared_group, shared_num_otus]
            shared_line_columns.extend(otus)
            # make the column width 10 use {:10}
            shared_line = '\t'.join(['{}'.format(column) for column in shared_line_columns])
            write_and_print(shared_line, iris_shared_file)

            ##design_group = '{}.{}'.format(iris_target_names[target],i)
            design_treatment = '{}'.format(iris_target_names[target])
            design_line_columns = [shared_group, design_treatment]
            design_line = '\t'.join(['{}'.format(column) for column in design_line_columns])
            write_and_print(design_line, iris_design_file)


def write_and_print(line, file):
    print(line)
    file.write(line)
    file.write('\n')

if __name__ == '__main__':
    make_files()