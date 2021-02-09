import numpy as np


def lwcov(X):
    nobs, nvar = X.shape[0], X.shape[1]
    X = X-np.average(X, axis=0)
    S = (1/(nobs-1))*np.matmul(np.transpose(X), X)
    m = np.trace(S)/nvar
    d2 = np.square(np.linalg.norm(S-m*np.eye(nvar)))/nvar
    rownorms = np.sum(np.square(X), axis=1)
    term11 = np.matmul(np.transpose(X), np.multiply(np.transpose(X), rownorms))
    term12 = -2*np.matmul(S, (np.matmul(np.transpose(X), X)))
    term22 = nobs*np.matmul(S, np.transpose(S))
    b2 = np.trace(term11+term12+term22)/(nvar*nobs^2)
    a2 = d2 - b2
    Sr = b2/d2 * m * np.eye(nvar)+a2/d2*S

    return Sr
