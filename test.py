import functionE
import CyfunctionE 
import numpy as np
import time

def execute(D, N, X, beta, tetha):
        initial = time.time()
        functionE.rbf_network(X, beta, tetha)
        tiempoPy = time.time() - initial

        initial = time.time()
        CyfunctionE.rbf_network(X, beta, tetha)
        tiempoCy = time.time() - initial

        SpeedUp = round(tiempoPy/tiempoCy, 3)

        print("tiempo Python: {}\n".format(tiempoPy))
        print("tiempo Cython: {}\n".format(tiempoCy))
        print("tiempo SpeedUp: {}\n".format(SpeedUp))

N = 1500
beta = np.random.rand(N)
tetha = 10

D = 5
X = np.array([np.random.rand(N) for d in range(D)]).T
execute(D, N, X, beta, tetha)

D = 50
X = np.array([np.random.rand(N) for d in range(D)]).T
execute(D, N, X, beta, tetha)
