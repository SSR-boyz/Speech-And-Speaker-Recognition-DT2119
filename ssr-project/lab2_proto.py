import numpy as np
from lab2_tools import *
from prondict import prondict
import matplotlib.pyplot as plt

def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
    
    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
            means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    M1 = np.size(hmm1['startprob']) - 1
    M2 = np.size(hmm2['startprob']) - 1

    hmm3 = {}
    hmm3['startprob'] = hmm2['startprob'] * hmm1['startprob'][M1]
    hmm3['startprob'] = np.concatenate((hmm1['startprob'][0:M1], hmm3['startprob']))

    mul = np.reshape(hmm1['transmat'][0:-1, -1], (M1, 1)) @ np.reshape(hmm2['startprob'], (1, M2+1))
    hmm3['transmat'] =  np.concatenate((hmm1['transmat'][0:-1, 0:-1], mul), axis=1)

    tmp = np.concatenate((np.zeros([M2+1,M1]), hmm2['transmat']), axis=1)
    hmm3['transmat'] = np.concatenate((hmm3['transmat'], tmp), axis=0)

    hmm3['means'] = np.concatenate((hmm1['means'], hmm2['means']), axis=0)
    hmm3['covars'] = np.concatenate((hmm1['covars'], hmm2['covars']), axis=0)
    
    return hmm3

# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """
    

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    def alphas(n, a, log_emlik, log_startprob, log_transmat):
        if n == 0:
            a[0] = np.log(log_startprob) + log_emlik[0]
            return a[0]
        else:
            a[n-1] = alphas(n-1, a, log_emlik, log_startprob, log_transmat)
            
            _,M = np.shape(log_emlik)
            for j in range(0, M):
                a[n][j] = logsumexp(a[n-1] + np.log(log_transmat[:,j])) + log_emlik[n][j]
            return a[n]


    alpha = np.zeros(np.shape(log_emlik))
    N, M = np.shape(log_emlik)
    alphas(N-1, alpha, log_emlik, log_startprob, log_transmat)

    return alpha, logsumexp(alpha[N-1])

def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    def betas(n, beta, log_emlik, log_startprob, log_transmat):
        N, M = log_emlik.shape
        if n == N - 1:
            beta[N-1] = np.zeros([1,M])
            return beta[N-1]
        else:
            beta[n+1] = betas(n+1, beta, log_emlik, log_startprob, log_transmat)
            for i in range(0, M):

                beta[n][i] = logsumexp( np.log(log_transmat[i, :]) + log_emlik[n+1] + beta[n+1] ) 
                
            return beta[n]
    
    beta = np.zeros(np.shape(log_emlik))
    N, M = np.shape(log_emlik)
    betas(0, beta, log_emlik, log_startprob, log_transmat)
    
    return beta, logsumexp(beta[0])    

def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    N, M = log_emlik.shape
    B = np.zeros((N, M))
    V = np.zeros((N, M))
    V[0] = np.log(log_startprob) + log_emlik[0]

    for t in range(1, N):
        for j in range(M):
            V[t][j] = np.max(V[t-1] + np.log(log_transmat[:,j])) + log_emlik[t][j]
            B[t][j] = np.argmax(V[t-1] + np.log(log_transmat[:,j]))
    
    best_score = np.max(V[t])
    
    path = np.zeros(N)
    path[t] = np.argmax(B[t, :])

    for t in range(N-2, 0, -1):
        path[t] = B[t+1,int(path[t+1])]
    
    return path, best_score

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    N, M = log_alpha.shape
    log_gamma = np.zeros(np.shape(log_alpha))
    
    for n in range(N):
        log_gamma[n] = log_alpha[n] + log_beta[n] - logsumexp(log_alpha[N-1])    
    
    return log_gamma


def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
         were N is the lenght of the observation sequence, D is the
         dimensionality of the feature vectors and M is the number of
         states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    N, M = log_gamma.shape
    D = X.shape[1]
    gamma = np.exp(log_gamma)

    # Means
    means = gamma.T @ X
    denominator = np.sum(gamma, axis=0).reshape(-1, 1)
    means = np.divide(means, denominator)

    # Covars
    covars = np.zeros([M,D])
    for state in range(M):
        nom = 0
        for timestep in range(N):
            #print(np.shape((X[timestep, :] - means[state, :])))
            nom += gamma[timestep, state] * ((X[timestep, :] - means[state, :]).reshape(-1,1) @ (X[timestep, :]  - means[state, :]).reshape(-1,1).T)
        covars[state] = np.divide(nom.diagonal(), denominator[state])

    covars[np.where(covars < varianceFloor)] = varianceFloor
    return means, covars
    

def baum_welch(data, wordHMM, settings):
    """ 
    1. Expectation: compute the alpha, beta and gamma probabilities for the utterance, given the
       current model parameters (using your functions forward(), backward() and statePosteriors())
    
    2. Maximization: update the means µjk and variances σ ik given the sequence of feature vectors
       and the gamma probabilities (using updateMeanAndVar())
        
    3. estimate the likelihood of the data given the new model, if the likelihood has increased, go
       back to 1
    """
    old_lik = np.NINF
    likelihood = 0
    for it in range(settings['max_iter']):
        obsloglik = log_multivariate_normal_density_diag(data, wordHMM['means'], wordHMM['covars'])
        alpha, likelihood = forward(obsloglik, wordHMM['startprob'][:-1], wordHMM['transmat'][:-1, :-1])
        beta, likelihood_beta = backward(obsloglik, wordHMM['startprob'][:-1], wordHMM['transmat'][:-1, :-1])
        gamma = statePosteriors(alpha, beta)
        wordHMM['means'], wordHMM['covars'] = updateMeanAndVar(data, gamma)

        if np.isnan(likelihood):
            likelihood = old_lik
            break

        if old_lik + settings['threshold'] > likelihood:
            likelihood = old_lik
            break

        old_lik = likelihood
    
    return likelihood, it+1


