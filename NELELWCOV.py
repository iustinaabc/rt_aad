import numpy as np
from sklearn import covariance

'''
def lwcov(X):
    nobs, nvar = X.shape[0], X.shape[1] # 24, 7200
    X = X-np.average(X, axis=0)
    S = (1/(nobs-1))*np.matmul(X,np.transpose(X))
    print("shape S", np.shape(S))
    m = np.trace(S)/nvar
    print(m)
    d2 = np.square(np.linalg.norm(S-m*np.eye(nobs)))/nvar
    rownorms = np.sum(np.square(X), axis=1)
    print("shape roxnorms", np.shape(rownorms))
    print("np.multiply(np.transpose(X),rownorms)", np.shape(np.multiply(np.transpose(X),rownorms)))
    print("np.transpose(X)", np.shape(np.transpose(X)))
    term11 = np.matmul(X, np.multiply(np.transpose(X), rownorms))
    # term11 = np.matmul(np.transpose(np.multiply(np.transpose(X),rownorms)), np.transpose(X))
    print("shape term11", np.shape(term11))
    term12 = -2*np.matmul(S, (np.matmul(X, np.transpose(X))))
    term22 = nobs*np.matmul(S, np.transpose(S))
    b2 = np.trace(term11+term12+term22)/((nvar*nobs)^2)
    a2 = d2 - b2
    Sr = b2/d2 * m * np.eye(nobs)+a2/d2*S
    print("shape Sr", np.shape(Sr))

    return Sr
'''


def lwcov(X):
    print("----- X -----")
    print(X)
    nobs, nvar = X.shape[0], X.shape[1]   # n_samples, n_features
    print("nobs", nobs)
    print("nvar", nvar)
    X = X-np.average(X, axis=0)
    print("----- X avg-----")
    print(X)
    cov = (1/(nobs-1))*np.matmul(np.transpose(X), X) # (X.T * X) / (nobs - 1)  # np.cov(X)
    print("----- cov -----")
    print(cov)
    # cov = (cov + np.transpose(cov)) / 2

    delta_ = np.trace(np.matmul(np.transpose(X), X))
    print("delta_", delta_)

    X2 = X ** 2
    beta_ = np.trace(np.matmul(X2, np.transpose(X2)))
    print("beta_:", beta_)

    mu = np.trace(cov)/nvar
    delta = np.linalg.norm(cov - mu * np.eye(nvar)) / nvar
    # grootste eigenwaarde van [cov - mu*eye]STER * [cov - mu*eye]
    # of dus (eigenwaarde van [cov - mu*eye] )^2
    # print("---info---")
    # print(np.linalg.norm(cov - mu * np.eye(nvar)))
    rownorms = np.sum(np.square(X), axis=1) / nobs

    # x11 * x11 + ....
    term11 = np.matmul(np.transpose(X), np.transpose(np.multiply(np.transpose(X), rownorms)))

    # x12 * x21 + ....
    term12 = -2*np.matmul(cov, (np.matmul(np.transpose(X), X))) #(X.T * X)^2

    term22 = nobs*np.matmul(cov, np.transpose(cov)) # (X.T * X) * (X.T * X).T

    b2 = np.trace(term11+term12+term22)/(nvar*(nobs)**2)
    print("b2", b2)
    beta = 1 / (nobs * nvar)  * ( beta_ / nobs - delta_ )
    print("beta", beta)

    beta2 = min(beta, delta)
    print("beta2", beta2)
    if delta == 0:
        shrinkage = 0
    else:
        shrinkage = beta2 / delta
    #a2 = d2 - b2
    #shrinkage = lw_shrinkage(X)
    shrunk_cov = (1-shrinkage) * cov + shrinkage * mu * np.eye(nvar)
    #

    return shrunk_cov
    
'''


def lw_shrinkage(X):
    pass


def lwcov(X):
    nobs, nvar = X.shape[0], X.shape[1]
    X = X-np.average(X, axis=0)
    S = (1/(nobs-1))*np.matmul(np.transpose(X), X)
    #S = (S + np.transpose(S)) / 2
    m = np.trace(S)/nvar
    d2 = np.square(np.linalg.norm(S-m*np.eye(nvar)))/nvar
    rownorms = np.sum(np.square(X), axis=1)
    term11 = np.matmul(np.transpose(X), np.transpose(np.multiply(np.transpose(X), rownorms)))
    term12 = -2*np.matmul(S, (np.matmul(np.transpose(X), X)))
    term22 = nobs*np.matmul(S, np.transpose(S))
    b2 = np.trace(term11+term12+term22)/((nvar*nobs)^2)
    a2 = d2 - b2
    Sr = b2/d2 * m * np.eye(nvar) + a2/d2*S

    return Sr
'''
def main():
    matrix = np.array([ [1, 5, 6, 4], [9 , 10, 8, 7], [5, 10 ,6 ,8] ])
    cov1 = np.cov(matrix)
    cov2 = lwcov(np.transpose(matrix))
    cov3 = covariance.ledoit_wolf(np.transpose(matrix))[0]

    print("------matrix------")
    print(matrix)
    print("------np.cov------")
    print(cov1)
    print("------lwcov------")
    print(cov2)
    print("------ledoit_wolf------")
    print(cov3)

main()