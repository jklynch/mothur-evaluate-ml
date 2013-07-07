import numpy as np

import smo

def test_smo():
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

    labels = []
    labels.append('blue')
    labels.append('blue')
    labels.append('blue')
    labels.append('blue')
    labels.append('green')
    labels.append('green')
    labels.append('green')
    labels.append('green')

    smo.smo(x, labels)
