#!/usr/bin/env python3
"""
@author: Nele Eeckman, Sofie Mareels
"""
import numpy as np
from tmprod import tmprod
import scipy
from lwcov import lwcov
from group_by_class import group_by_class
import scipy.linalg as la

def CSP(class_covariances, size):
    # Solve the generalized eigenvalue problem resulting in eigenvalues and corresponding eigenvectors and
    # sort them in descending order.
    eigenvalues, eigenvectors = la.eigh(class_covariances[0], class_covariances[1])
    id_descending = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, id_descending]
    eigenvectors_begin = np.array(eigenvectors[:, :int(size/2)])
    eigenvectors_end = np.array(eigenvectors[:, int(np.shape(eigenvectors)[1]-size/2):])
    eigenvectors = np.concatenate((eigenvectors_begin, eigenvectors_end), axis=1)
    return eigenvectors


def trainCSP(X, y, spatial_dim, optmode, covMethod='lwcov'):
    """
    Trains the FBCSP filters on the supplied EEG data and labels

    Parameters
    ----------
    :param X: (4-dimensional numpy array) The EEG to train the filters on
                                          Dimensions should be: channels x band x time x trial

    :param y: (1-dimensional numpy array) The labels defining a class for each trial

    :param spatial_dim: (int) Amount of Spatial filters used per band (K in paper)

    :param optmode: (str) optimization mode: 'ratiotrace' or 'traceratio'

    :param covMethod: (str) covariance matrix estimation method: 'classic' / 'lwcov'

    :return: W: (2-dimensional numpy array) CSP filters to maximize output energy
                                            Dimensions should be: nr. of filter x channel
             score: (1-dimensional numpy array) The score of the CSP filter in W
             tr: (1-dimensional numpy array) The traceratio of the filtered EEG for each class

    """

    # Initialize
    if optmode != 'ratiotrace':
        optmode = 'ratiotrace'
    # Divide into X for each class
    X1, X2 = group_by_class(X, y)
    X1 = np.transpose(X1, (0, 2, 1))
    X2 = np.transpose(X2, (0, 2, 1))

    # TODO: lwcov doesn't work, need to look at it
    if covMethod == 'lwcov':
        first = True
        S = []
        for group in [X1, X2]:
            for trials in group:
                if first:
                    s_temp = lwcov(trials)
                    first = False
                else:
                    s_temp += lwcov(trials)
            first = True
            s_temp = s_temp / np.shape(group)[0]
            S.append(s_temp)
    elif covMethod == 'classic':
        first = True
        S = []
        for group in [X1, X2]:
            for trials in group:
                if first:
                    Stemp = np.cov(trials)
                    first = False
                else:
                    Stemp += np.cov(trials)
            first = True
            Stemp = Stemp / np.shape(group)[0]
            S.append(Stemp)


    '''---Optimize CSP filters---'''

    # Onze code

    W = CSP(S, spatial_dim)

    # Oude code, niet goed
    '''
    S1 = S[0]
    S2 = S[1]
    
    if optmode == 'ratiotrace':
        if spatial_dim is None:
            patidx = list(range(X.shape[0])) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            print("patidx NONE", patidx)
        else:
            patidx = list(range(int(np.ceil(spatial_dim / 2)))) + list(range(X.shape[0] - int(np.ceil(spatial_dim / 2)), X.shape[0]))
            print("patidx ELSE",patidx)
        
        S1 = S1 / (np.trace(S1))
        S2 = S2 / (np.trace(S2))
        
        D, W = scipy.linalg.eig(S1+0.01*np.diag(np.diag(S1)), S2+0.01*np.diag(np.diag(S2)))
        # D, W = np.linalg.eig(np.matmul(S1, np.linalg.inv(S2)))
        # D,W = np.linalg.eig(np.matmul(S1,S1+S2))
        lamda = D
        print(lamda.shape)

        if True:
            print("X1 SHAPE", X1.shape)
            print('W SHAPE', W.shape)
            Y1 = tmprod(X1, np.transpose(W))
            Y2 = tmprod(X2, np.transpose(W))
            print("SHAPE Y2", Y2.shape)
            Y1 = np.var(Y1, axis=1)
            Y2 = np.var(Y2, axis=1)
            score = np.median(Y1, axis=1) / (np.median(Y1, axis=1) + np.median(Y2, axis=1))
        else:
            score = lamda

        # sort according to median relation
        order = np.argsort(-score)
        score = -np.sort(-score)
        lamda = lamda[order]
        W = W[:, order]

        # Truncate to the desired number of CSP patterns
        W = W[:, patidx]
        lamda = lamda[patidx]
        score = score[patidx]
        tr = np.zeros((2,))
        # tr[0] = np.trace(np.transpose(W)*S1*W)/np.trace(np.transpose(W)*(S1+S2)*W)
        tr[0] = np.trace(np.matmul(np.matmul(np.transpose(W), S1), W)) / \
                np.trace(np.matmul(np.matmul(np.transpose(W), (S1 + S2)), W))
        # tr[1] = np.trace(np.transpose(W)*S2*W)/np.trace(np.transpose(W)*(S1+S2)*W)
        tr[1] = np.trace(np.matmul(np.matmul(np.transpose(W), S2), W)) / \
                np.trace(np.matmul(np.matmul(np.transpose(W), (S1 + S2)), W))
    '''

    '''
    elif optmode == 'traceratio':

        # Initialize
        spatial_dimhalf = round(spatial_dim / 2)

        # compute CSP filters for class 1 normalized vector basis for that dimension
        W1 = np.random.randn(S1.shape[0], spatial_dimhalf)
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
            labda = labda[0:spatial_dimhalf]
            W1 = temp[:, 0:spatial_dimhalf]
            relchange = abs(tr - tr0) / tr
            tr0 = tr

        score = labda

        # compute CSP filters for class 2
        W2 = np.random.randn(S2.shape[0], spatial_dimhalf)
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
            labda = labda[0:spatial_dimhalf]
            W2 = temp[:, :spatial_dimhalf]
            relchange = abs(tr - tr0) / tr
            tr0 = tr

        score = np.concatenate((score, labda))
        W = np.concatenate((W1, W2), axis=1)

        tr = np.zeros((2))
        tr[0] = np.trace(np.matmul(np.matmul(np.transpose(W1), S1), W1)) / \
                np.trace(np.matmul(np.matmul(np.transpose(W1), (S1 + S2)), W1))
        tr[1] = np.trace(np.matmul(np.matmul(np.transpose(W2), S2), W2)) / \
                np.trace(np.matmul(np.matmul(np.transpose(W2), (S1 + S2)), W2))
    '''

    score = 0
    tr = [0, 0]

    return W, score, tr
