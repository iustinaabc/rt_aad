import numpy as np
from tmprod import tmprod
import scipy
from lwcov import lwcov

def trainCSP(X, y, npat, optmode, covMethod='lwcov'):
    """
    Trains the FBCSP filters on the supplied EEG data and labels

    Parameters
    ----------
    :param X: (4-dimensional numpy array) The EEG to train the filters on
                                          Dimensions should be: channels x band x time x trial

    :param y: (1-dimensional numpy array) The labels defining a class for each trial

    :param npat: (int) Amount of Spatial filters used per band (K in paper)

    :param optmode: (str) optimization mode: 'ratiotrace' or 'traceratio'

    :param covMethod: (str) covariance matrix estimation method: 'classic' / 'lwcov'

    :return: W: (2-dimensional numpy array) CSP filters to maximize output energy
                                            Dimensions should be: nr. of filter x channel
             score: (1-dimensional numpy array) The score of the CSP filter in W
             tr: (1-dimensional numpy array) The traceratio of the filtered EEG for each class
    """

    # Initialize
    if optmode != 'traceratio' and optmode != 'ratiotrace':
        optmode = 'ratiotrace'
    # Divide into X for each class
    yc = np.unique(y)
    indices1 = np.where(y == yc[0])
    indices2 = np.where(y == yc[1])

    X1 = X[:, :, indices1[0]]
    X2 = X[:, :, indices2[0]]
    
    Xm1 = np.reshape(X1, (X1.shape[0], X1.shape[1] * X1.shape[2]))
    Xm2 = np.reshape(X2, (X2.shape[0], X2.shape[1] * X2.shape[2]))

    if covMethod == 'lwcov':
        S1 = lwcov(Xm1)
        S2 = lwcov(Xm2)
    elif covMethod == 'classic':
        S1 = np.cov(Xm1)
        S2 = np.cov(Xm2)

    if npat is None:
        patidx = list(range(X.shape[0]))
    else:
        patidx = list(range(int(np.ceil(npat / 2)))) + list(range(X.shape[0] - int(np.ceil(npat / 2)), X.shape[0]))

    S1 = S1 / (np.trace(S1))
    S2 = S2 / (np.trace(S2))

    # Optimize CSP filters
    if optmode == 'ratiotrace':
        D, W = scipy.linalg.eig(S1+0.01*np.diag(np.diag(S1)), S2+0.01*np.diag(np.diag(S2)))
        # D, W = np.linalg.eig(np.matmul(S1, np.linalg.inv(S2)))
        # D,W = np.linalg.eig(np.matmul(S1,S1+S2))
        labda = D
        print(labda.shape)

        if True:
            print(X1.shape, 'bam', W.shape)
            Y1 = tmprod(X1, np.transpose(W))
            Y2 = tmprod(X2, np.transpose(W))
            print(Y2.shape)
            Y1 = np.var(Y1, axis=1)
            Y2 = np.var(Y2, axis=1)
            score = np.median(Y1, axis=1) / (np.median(Y1, axis=1) + np.median(Y2, axis=1))
        else:
            score = labda

        # sort according to median relation
        order = np.argsort(-score)
        score = -np.sort(-score)
        labda = labda[order]
        W = W[:, order]

        # Truncate to the desired number of CSP patterns
        W = W[:, patidx]
        labda = labda[patidx]
        score = score[patidx]
        tr = np.zeros((2,))
        # tr[0] = np.trace(np.transpose(W)*S1*W)/np.trace(np.transpose(W)*(S1+S2)*W)
        tr[0] = np.trace(np.matmul(np.matmul(np.transpose(W), S1), W)) / \
                np.trace(np.matmul(np.matmul(np.transpose(W), (S1 + S2)), W))
        # tr[1] = np.trace(np.transpose(W)*S2*W)/np.trace(np.transpose(W)*(S1+S2)*W)
        tr[1] = np.trace(np.matmul(np.matmul(np.transpose(W), S2), W)) / \
                np.trace(np.matmul(np.matmul(np.transpose(W), (S1 + S2)), W))
    elif optmode == 'traceratio':

        # Initialize
        npathalf = round(npat / 2)

        # compute CSP filters for class 1 normalized vector basis for that dimension
        W1 = np.random.randn(S1.shape[0], npathalf)
        W1, r = np.linalg.qr(W1)

        relchange = np.Inf
        tr0 = np.Inf
        # Loop until the K/2 best filters are chosen
        # W shows the vectors where to transform to
        while relchange > 1e-3:
            # print(W1.shape)
            tr = np.trace(np.matmul(np.matmul(np.transpose(W1), S1), W1)) / \
                 np.trace(np.matmul(np.matmul(np.transpose(W1), (S1 + S2)), W1))

            # this will make sure energy after filter is big because tr shows which
            labda, temp = np.linalg.eig(S1 - tr * (S1 + S2))

            order = np.argsort(-labda)
            labda = -np.sort(-labda)
            temp = temp[:, order]
            labda = labda[0:npathalf]
            W1 = temp[:, 0:npathalf]
            relchange = abs(tr - tr0) / tr
            tr0 = tr

        score = labda

        # compute CSP filters for class 2
        W2 = np.random.randn(S2.shape[0], npathalf)
        W2, r = np.linalg.qr(W2)

        relchange = np.Inf
        tr0 = np.Inf
        while relchange < 1e-3:
            tr = np.trace(np.matmul(np.matmul(np.transpose(W2), S2), W2)) / \
                 np.trace(np.matmul(np.matmul(np.transpose(W2), (S1 + S2)), W2))
            labda, temp = np.linalg.eig(S2 - tr * (S1 + S2))
            order = np.argsort(-labda)
            labda = -np.sort(-labda)
            temp = temp[:, order]
            labda = labda[0:npathalf]
            W2 = temp[:, :npathalf]
            relchange = abs(tr - tr0) / tr
            tr0 = tr

        score = np.concatenate((score, labda))
        W = np.concatenate((W1, W2), axis=1)

        tr = np.zeros((2))
        tr[0] = np.trace(np.matmul(np.matmul(np.transpose(W1), S1), W1)) / \
                np.trace(np.matmul(np.matmul(np.transpose(W1), (S1 + S2)), W1))
        tr[1] = np.trace(np.matmul(np.matmul(np.transpose(W2), S2), W2)) / \
                np.trace(np.matmul(np.matmul(np.transpose(W2), (S1 + S2)), W2))

    return W, score, tr
