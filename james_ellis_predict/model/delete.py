import numpy as np
def diff(val1, e1, val2, e2):
    e = np.sqrt(e1**2 + e2**2)
    print('Diff: {}, error: {:.2f}'.format(val2-val1, e))

diff(43.9, 0.1, 44.41, 0.07)
diff(43.9, 0.1, 44.58, 0.05)



