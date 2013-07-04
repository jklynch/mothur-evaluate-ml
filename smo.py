
def smo():
    print('hazzah!')

    # we start out with this trivial data
    x_01 = (0.0, 0.0)
    x_02 = (0.0, 1.0)
    x_21 = (2.0, 0.0)
    x_22 = (2.0, 1.0)
    labels = []
    labels.append('blue')
    labels.append('green')
    labels.append('blue')
    labels.append('green')
    x = []
    x.append(x_01)
    x.append(x_21)
    x.append(x_02)
    x.append(x_22)
    #

    # here is some different trivial data
    x = []
    x.append((1.0, 3.0))
    x.append((2.0, 5.0))
    x.append((3.0, 8.0))
    x.append((6.0, 4.0))

    x.append((6.0, 7.0))
    x.append((7.0, 8.0))
    x.append((8.0, 4.0))
    x.append((3.0, 6.0))

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
    y = [label_class[label_i] for label_i in labels]
    print('i x y label')
    for i in range(len(labels)):
        print('{} {} {} {}'.format(i, x[i], y[i], labels[i]))

    # find K
    K = []
    for (i, x_i) in enumerate(x):
        K_i = []
        for (j, x_j) in enumerate(x):
            #print('i = {} x_i = {}'.format(i, x_i))
            #print('j = {} x_j = {}'.format(j, x_j))
            #K[i][j] = sum( [xik * xjk for xik, xjk in zip(x_i, x_j)] )
            K_i.append(sum( [xik * xjk for xik, xjk in zip(x_i, x_j)] ))
        K.append(K_i)
        print('K = {}'.format(K))
    for i in range(len(K)):
        print('K[{}][:]={}'.format(i, K[i]))

    # begin SMO
    # need A and B
    A = len(y)*[0.0]
    B = len(y)*[0.0]
    C = 1.0
    for i in range(len(labels)):
        if y[i] == +1.0:
            A[i], B[i] = 0.0, C
        else:
            A[i], B[i] = -C, 0.0

    # initialize a and g
    a = len(labels)*[0.0]
    g = len(labels)*[1.0]
    n = 0
    while (True):
        n += 1
        print('n = {}'.format(n))
        i = None #0
        j = None #0
        yg_max = float('-Inf')
        yg_min = float('+Inf')
        print('A : {}'.format(A))
        print('B : {}'.format(B))
        print('y : {}'.format(y))
        print('a : {}'.format(a))
        print('g : {}'.format(g))
        # print ya and yg
        ya = [yk*ak for (yk, ak) in zip(y, a)]
        print('ya: {}'.format(ya))
        yg = [yk*gk for (yk, gk) in zip(y, g)]
        print('yg: {}'.format(yg))
        for k in range(len(y)):
            print('k = {}'.format(k))
            print('  ya[{}] < B[{}]: {} < {} : {}'.format(k, k, ya[k], B[k], ya[k] < B[k]))
            if ya[k] < B[k]:
                print('x[{}] = {} in I_up'.format(k, x[k]))
                print('    yg[{}] > yg_max: {} > {} : {}'.format(k, yg[k], yg_max, yg[k] > yg_max))
                if yg[k] > yg_max: #yg[i]:
                    yg_max = yg[k]
                    i = k
                    print('      i = {}'.format(k))
            print('  A[{}] < ya[{}]: {} < {} : {}'.format(k, k, A[k], ya[k], A[k] < ya[k]))
            if A[k] < ya[k]:
                print('x[{}] = {} in I_down'.format(k, x[k]))
                print('    yg[{}] < yg_min: {} < {} : {}'.format(k, yg[k], yg_min, yg[k] < yg_min))
                if yg[k] < yg_min: #< yg[j]:
                    print('      j = {}'.format(k))
                    yg_min = yg[k]
                    j = k
        print('i = {} j = {}'.format(i, j))
        print('maximal violating pair:')
        print('  i = {} x[{}] = {}'.format(i, i, x[i]))
        print('  j = {} x[{}] = {}'.format(j, j, x[j]))

        #if n > 3:
        #  break

        print('checking optimality criterion')
        print('  yg[{}] <= yg[{}]: {} <= {} {}'.format(i, j, yg[i], yg[j], yg[i] <= yg[j]))
        if yg[i] <= yg[j]:
            print('optimality criterion has been met')
            break

        # direction search - important to eliminate 0.0 from consideration for lambda
        u = []
        u_i = B[i] - ya[i]
        if u_i > 0.0:
            u.append(u_i)
        u_j = ya[j] - A[j]
        if u_j > 0.0:
            u.append(u_j)
        u_ij = (yg[i]-yg[j]) / (K[i][i] + K[j][j] - 2.0 * K[i][j])
        if u_ij > 0.0:
            u.append(u_ij)
        print('(yg[{}] - yg[{}])/(K[{}][{}] + K[{}][{}] - 2.0 * K[{}][{}])'.format(i, j, i, i, j, j, i, j))
        print('({} - {})/({} + {} - 2.0 * {})'.format(yg[i], yg[j], K[i][i], K[j][j], K[i][j]))
        print('{}'.format((yg[i]-yg[j]) / (K[i][i] + K[j][j] - 2.0 * K[i][j])))
        #u.append( (yg[i]-yg[j]) / (K[i][i] + K[j][j] - 2.0 * K[i][j]) )
        print('directions: {}'.format(u))
        l = min(u)
        print('lambda = {}'.format(l))

        # update gradient
        print('K[{}] = {}'.format(i, K[i]))
        print('K[{}] = {}'.format(j, K[j]))
        for k in range(len(g)):
            #g[k] = g[k] - l*y[k]*K[i][k] + l*y[k]*K[j][k]
            g[k] += (-l*y[k]*K[i][k] + l*y[k]*K[j][k])
        # update coefficients
        a[i] = a[i] + l * y[i]
        a[j] = a[j] - l * y[j]

    # find the optimal hyperplane
    w = len(x[0]) * [0.0,]
    b = None
    for A_i, B_i, a_i, y_i, yg_i, x_i in zip(A, B, a, y, yg, x):
        print('x_i = {}, y_i = {}, a_i = {}, yg_i = {}'.format(x_i, y_i, a_i, yg_i))
        if A_i < a_i < B_i:
            b = yg_i
        for j, x_ij in enumerate(x_i):
            w[j] += a_i * y_i * x_ij
    print('w* = {} b* = {}'.format(w, b))


if __name__ == '__main__':
    smo()
