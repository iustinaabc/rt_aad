import numpy as np

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
