import numpy as np
import pylab as pl

def smo():
    print('hazzah!')

    # here is some trivial data
    x = np.array([[1.0, 3.0],
             [2.0, 5.0],
             [3.0, 8.0],
             [6.0, 4.0],
             [6.0, 7.0],
             [7.0, 8.0],
             [8.0, 4.0],
             [3.0, 6.0]])

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x = (x - mean) / std

    labels = []
    labels.append('blue')
    labels.append('blue')
    labels.append('blue')
    labels.append('blue')
    labels.append('green')
    labels.append('green')
    labels.append('green')
    labels.append('green')

    # we need to assign numeric class labels -1 and +1
    all_labels = list({label_i for label_i in labels})
    all_labels.sort()
    label_class = {}
    if len(all_labels) == 2:
        for label_i in all_labels:
            print('label: {}'.format(label_i))
        label_class[all_labels[0]] = -1.0
        label_class[all_labels[1]] = +1.0
    else:
        print('too many labels: '.format(all_labels))

    # the y column vector gives the class label (-1.0 or +1.0)
    # for each row of x, so y looks like:
    #   y = [-1.0 -1.0 ... 1.0 1.0].transpose()
    y = np.array([label_class[label_i] for label_i in labels]).transpose()
    print('y = \n{}'.format(y))
    print('i x y label')
    for i in range(len(labels)):
        print('{} {} {} {}'.format(i, x[i,:], y[i], labels[i]))

    # find K
    # x is observations-by-features so we want x*x'
    K = np.dot(x, x.T)
    print('K = \n{}'.format(K))

    # begin SMO
    # need A and B
    A = np.zeros(y.shape)
    print('A = {}'.format(A))
    B = np.zeros(y.shape)
    C = 8.0
    for i in range(len(labels)):
        if y[i] == +1.0:
            A[i], B[i] = 0.0, C
        else:
            A[i], B[i] = -C, 0.0

    # initialize a and g
    a = np.zeros(y.shape[0])
    g = np.ones(y.shape[0])
    n = 0
    while (True):
        n += 1
        print('n = {}'.format(n))
        i = None #0
        j = None #0
        yg_max = float('-Inf')
        yg_min = float('+Inf')
        print("A' = {}".format(A.T))
        print("B' = {}".format(B.T))
        print("y' = {}".format(y.T))
        print("a' = {}".format(a.T))
        print("g' = {}".format(g.T))
        # print ya and yg
        ya = y * a
        print("ya' = {}".format(ya.T))
        yg = y * g
        print("yg' = {}".format(yg.T))
        for k in range(len(y)):
            print('k = {}'.format(k))
            print('  ya[{}] < B[{}]: {} < {} : {}'.format(k, k, ya[k], B[k], ya[k] < B[k]))
            if ya[k] < B[k]:
                print('x[{}] = {} in I_up'.format(k, x[k]))
                print('    yg[{}] > yg_max: {} > {} : {}'.format(k, yg[k], yg_max, yg[k] > yg_max))
                if yg[k] > yg_max:
                    yg_max = yg[k]
                    i = k
                    print('      i = {}'.format(k))
            print('  A[{}] < ya[{}]: {} < {} : {}'.format(k, k, A[k], ya[k], A[k] < ya[k]))
            if A[k] < ya[k]:
                print('x[{}] = {} in I_down'.format(k, x[k]))
                print('    yg[{}] < yg_min: {} < {} : {}'.format(k, yg[k], yg_min, yg[k] < yg_min))
                if yg[k] < yg_min:
                    print('      j = {}'.format(k))
                    yg_min = yg[k]
                    j = k
        print('i = {} j = {}'.format(i, j))
        print('maximal violating pair:')
        print('  i = {} x[{}] = {}'.format(i, i, x[i]))
        print('  j = {} x[{}] = {}'.format(j, j, x[j]))

        # what is a reasonable n to abort?
        #if n > 3:
        #  break

        print('checking optimality criterion')
        print('  yg[{}] <= yg[{}]: {} <= {} {}'.format(i, j, yg[i], yg[j], yg[i] <= yg[j]))
        if yg[i] <= yg[j]:
            print('optimality criterion has been met')
            break

        # direction search - important to eliminate 0.0 from consideration for lambda???
        u = []
        u_i = B[i] - ya[i]
        #if u_i > 0.0:
        u.append(u_i)
        u_j = ya[j] - A[j]
        #if u_j > 0.0:
        u.append(u_j)
        u_ij = (yg[i]-yg[j]) / (K[i,i] + K[j,j] - 2.0 * K[i,j])
        #if u_ij > 0.0:
        u.append(u_ij)
        print('directions: {}'.format(u))
        l = min(u)
        print('lambda = {}'.format(l))

        # update gradient
        print('K[{}] = {}'.format(i, K[i]))
        print('K[{}] = {}'.format(j, K[j]))
        for k in range(len(g)):
            g[k] += (-l*y[k]*K[i,k] + l*y[k]*K[j,k])
        # update coefficients
        a[i] = a[i] + l * y[i]
        a[j] = a[j] - l * y[j]

    # find the optimal hyperplane
    # the weight vector w is w = Sum( a_i*y_i*x_i )
    print('shape of x[0,:] is {}'.format(x[0,:].shape))
    w = np.zeros(x[0,:].shape)
    b = None
    for A_i, B_i, a_i, y_i, yg_i, x_i in zip(A, B, a, y, yg, x):
        print('x_i = {}, y_i = {}, a_i = {}, yg_i = {}'.format(x_i, y_i, a_i, yg_i))
        if A_i < a_i < B_i:
            b = yg_i
        w += a_i * y_i * x_i
    print('w* = {} b* = {}'.format(w, b))

    x_0_min, x_0_max = x[:,0].min() - 1, x[:,0].max() + 1
    x_1_min, x_1_max = x[:,1].min() - 1, x[:,1].max() + 1
    x_0 = np.arange(x_0_min, x_0_max, 0.01)
    x_1 = np.arange(x_1_min, x_1_max, 0.01)
    X_0, X_1 = np.meshgrid(x_0, x_1)
    Z = np.zeros(X_0.shape)
    for i in range(X_0.shape[0]):
        for j in range(X_0.shape[1]):
            Z[i,j] = np.sign(w[0]*X_0[i,j] + w[1]*X_1[i,j] + b)

    # decision surface
    pl.contourf(X_0, X_1, Z)
    # support vectors
    pl.scatter(x[a > 0.0,0], x[a > 0.0,1], s=100)
    # training data
    pl.scatter(x[:,0], x[:,1], s=50, c=labels)
    pl.show()


if __name__ == '__main__':
    smo()
