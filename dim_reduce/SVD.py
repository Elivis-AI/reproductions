import numpy as np
from sklearn.datasets import load_iris

def svd(data):
    '''
    :param data:
    :return: U, Sigma, VT
    '''

    # mean
    N, D = data.shape
    data = data - np.mean(data, axis=0)

    # V
    Veig_val, Veig_vector = np.linalg.eigh(np.dot(data.T, data))
    VT = Veig_vector[:, np.argsort(-abs(Veig_val))].T

    # U
    Ueig_val, Ueig_vector = np.linalg.eigh(np.dot(data, data.T))
    U = Ueig_vector[:, np.argsort(-abs(Ueig_val))]

    # Sigma
    Sigma = np.diag((data @ VT.T)[:D, np.arange(D)] / np.diag(U)[:D])
    # Sigma = np.zeros((N, D))
    # for i in range(D):
    #     Sigma[i, i] = np.dot(data, VT[i])[i]/U[i,i]

    return U, Sigma, VT

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target
    U, Sigma, VT = svd(X)
    # U_, Sigma_, VT_ = np.linalg.svd(X)
    # print(np.allclose(U, U_), np.allclose(Sigma, np.diag(Sigma_)), np.allclose(VT, VT_))

