# pylmm is a python-based linear mixed-model solver with applications to GWAS

# Copyright (C) 2014  Nicholas A. Furlotte (nick.furlotte@gmail.com)

# The program is free for academic use. Please contact Nick Furlotte
# <nick.furlotte@gmail.com> if you are interested in using the software for
# commercial purposes.

# The software must not be modified and distributed without prior
# permission of the author.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import time
import numpy as np
from scipy import linalg
from scipy import optimize
from scipy import stats

# np.seterr('raise')

def matrixMult(A, B):
    # If there is no fblas then we will revert to np.dot()
    try:
        linalg.fblas
    except AttributeError:
        return np.dot(A, B)

    # If the matrices are in Fortran order then the computations will be faster
    # when using dgemm.  Otherwise, the function will copy the matrix and that takes time.
    if not A.flags['F_CONTIGUOUS']:
        AA = A.T
        transA = True
    else:
        AA = A
        transA = False

    if not B.flags['F_CONTIGUOUS']:
        BB = B.T
        transB = True
    else:
        BB = B
        transB = False

    return linalg.fblas.dgemm(alpha=1., a=AA, b=BB, trans_a=transA, trans_b=transB)

def calculateKinshipIncremental(IN, numSNPs=None, computeSize=1000, center=False, missing="MAF"):
    """
    Uses lmm.calculateKinship to compute kinship matrices on input.plink objects using incremental
    version that reduces memory usage.
    :param IN: input.plink object initialized using Plink or EMMA formatted file
    :param numSNPs: The number of SNPs in the input.plink object, (Only needed if using EMMA file for input)
    :param computeSize: The maximum number of SNPs to read into memory at once (default 1000).
    :param center:
    :return:
    """
    n = len(IN.indivs)
    m = computeSize
    W = np.ones((n,m)) * np.nan

    IN.getSNPIterator()

    # Annoying hack to get around the fact that it is expensive to determine the number of SNPs in an emma file
    if numSNPs and IN.numSNPs != -1:
        print "numSNPs param not necessary when using plink files for input. Using numSNPs=" + str(IN.numSNPs)
    elif not numSNPs and IN.numSNPs < 0:
        raise Exception("numSNPs param is required when using EMMA files for input.")
    elif IN.numSNPs == -1:
        IN.numSNPs = numSNPs

    sys.stderr.write("Calculating kinship matrix for %d individuals\n" % n)
    sys.stderr.write("Total number of SNPs: %d\n" % IN.numSNPs)

    i = 0
    K = None
    while i < IN.numSNPs:
        j = 0
        while j < m and i < IN.numSNPs:
            snp,id = IN.next()

            if missing == "MAF":
                #calculate the mean of the values in this column that are not NaN
                mn = snp[True - np.isnan(snp)].mean()
                #replace all NaN values in this column with the mean
                snp[np.isnan(snp)] = mn

            vr = snp.var()
            if vr == 0:
                i += 1
                continue

            W[:,j] = snp

            i += 1
            j += 1

        if j < m:
            W = W[:,range(0,j)]

        sys.stderr.write("Processing first %d SNPs\n" % i)

        if K is None:
            K = matrixMult(W, W.T)
        else:
            K = K + matrixMult(W, W.T)

    K = K / float(IN.numSNPs)
    return K

def calculateEigendecomposition(K):
        return linalg.eigh(K)

def calculateKinship(W, center=False):
    """
    W is an n x m matrix encoding SNP minor alleles.
    This function takes a matrix oF SNPs, imputes missing values with the maf,
    normalizes the resulting vectors and returns the RRM matrix.
    """
    n = W.shape[0]
    m = W.shape[1]
    keep = []
    for i in range(m):
        #calculate the mean of the values in this column that are not NaN
        mn = W[True - np.isnan(W[:, i]), i].mean()
        #replace all NaN values in this column with the mean
        W[np.isnan(W[:, i]), i] = mn

        vr = W[:, i].var()
        if vr == 0:
            continue

        keep.append(i)
        W[:, i] = (W[:, i] - mn) / np.sqrt(vr)

    #keep contains all columns (SNPs) with non-zero variance, so this line removes all SNPs with zero variance prior to computing the kinship matrix
    W = W[:, keep]
    K = matrixMult(W, W.T) * 1.0 / float(m)
    if center:
        P = np.diag(np.repeat(1, n)) - 1 / float(n) * np.ones((n, n))
        S = np.trace(matrixMult(matrixMult(P, K), P))
        K_n = (n - 1) * K / S
        return K_n
    return K

def GWAS(Y, X, K, Kva=[], Kve=[], X0=None, REML=True, refit=False):
    """
        Performs a basic GWAS scan using the LMM.  This function
        uses the LMM module to assess association at each SNP and 
        does some simple cleanup, such as removing missing individuals 
        per SNP and re-computing the eigen-decomp
        Y - n x 1 phenotype vector
        X - n x m SNP matrix
        K - n x n kinship matrix
        Kva,Kve = linalg.eigh(K) - or the eigen vectors and values for K
        X0 - n x q covariate matrix
        REML - use restricted maximum likelihood
        refit - refit the variance component for each SNP
      """

    n = X.shape[0]
    m = X.shape[1]

    if X0 is None:
        X0 = np.ones((n, 1))

    # Remove missing values in Y and adjust associated parameters
    v = np.isnan(Y)
    if v.sum():
        keep = True - v
        keep = keep.reshape((-1,))
        Y = Y[keep]
        X = X[keep, :]
        X0 = X0[keep, :]
        K = K[keep, :][:, keep]
        Kva = []
        Kve = []

    if len(Y) == 0:
        return np.ones(m) * np.nan, np.ones(m) * np.nan

    L = LMM(Y, K, Kva, Kve, X0)
    if not refit:
        L.fit()

    PS = []
    TS = []

    n = X.shape[0]
    m = X.shape[1]

    for i in range(m):
        x = X[:, i].reshape((n, 1))
        v = np.isnan(x).reshape((-1,))
        if v.sum():
            keep = True - v
            xs = x[keep, :]
            if xs.var() == 0:
                PS.append(np.nan)
                TS.append(np.nan)
                continue

            Ys = Y[keep]
            X0s = X0[keep, :]
            Ks = K[keep, :][:, keep]
            Ls = LMM(Ys, Ks, X0=X0s)
            if refit:
                Ls.fit(X=xs)
            else:
                Ls.fit()
            ts, ps = Ls.association(xs, REML=REML)
        else:
            if x.var() == 0:
                PS.append(np.nan)
                TS.append(np.nan)
                continue

            if refit:
                L.fit(X=x)
            ts, ps = L.association(x, REML=REML)

        PS.append(ps)
        TS.append(ts)
    return TS, PS


def remove_missing_values(x, Y, X0, verbose, K, Kva, Kve, K2=None, K2va=None, K2ve=None):
    if not x.sum() == len(Y):
        if verbose:
            sys.stderr.write("Removing %d missing values from Y\n" % ((True - x).sum()))
        Y = Y[x]
        X0 = X0[x, :]
        K = K[x, :][:, x]

        if K2 is None:
            return x, Y, X0, K, [], []
        else:
            K2 = K2[x, :][:, x]
            return x, Y, X0, K, [], [], K2, [], []
    else:
        if K2 is None:
            return x, Y, X0, K, Kva, Kve
        else:
            return x, Y, X0, K, Kva, Kve, K2, K2va, K2ve


def do_eigendecomposition(K, K_name, verbose):
    if verbose:
        sys.stderr.write("Obtaining eigendecomposition for K1 (%dx%d matrix)\n" % (K.shape[0], K.shape[1]))
    begin = time.time()
    Kva, Kve = linalg.eigh(K)
    end = time.time()
    if verbose:
        sys.stderr.write("%s eigendecomposition time: %0.3f\n" % (K_name, end - begin))
    return Kva, Kve


def clean_eigenvalues(Kva, verbose):
    if verbose:
        sys.stderr.write("Cleaning %d eigenvalues\n" % (sum(Kva < 0)))
    Kva[Kva < 1e-6] = 1e-6
    return Kva


class LMM:
    """
        This is a simple version of EMMA/fastLMM.
        The main purpose of this module is to take a phenotype vector (Y), a set of covariates (X) and a kinship matrix (K)
        and to optimize this model by finding the maximum-likelihood estimates for the model parameters.
        There are three model parameters: heritability (h), covariate coefficients (beta) and the total
        phenotypic variance (sigma).
        Heritability as defined here is the proportion of the total variance (sigma) that is attributed to
        the kinship matrix.
        For simplicity, we assume that everything being input is a numpy array.
        If this is not the case, the module may throw an error as conversion from list to numpy array
        is not done consistently.
    """

    def __init__(self, Y, K, Kva=[], Kve=[], X0=None, verbose=False):
        """
        The constructor takes a phenotype vector or array of size n.
        It takes a kinship matrix of size n x n.  Kva and Kve can be computed as Kva,Kve = linalg.eigh(K) and cached.
        If they are not provided, the constructor will calculate them.
        X0 is an optional covariate matrix of size n x q, where there are q covariates.
        When this parameter is not provided, the constructor will set X0 to an n x 1 matrix of all ones to represent a mean effect.
        """
        if X0 is None:
            X0 = np.ones(len(Y)).reshape(len(Y), 1)
        self.verbose = verbose

        x = True - np.isnan(Y)
        x = x.reshape(-1, )
        if not x.sum() == len(Y):
            if self.verbose:
                sys.stderr.write("Removing %d missing values from Y\n" % ((True - x).sum()))
            Y = Y[x]
            K = K[x, :][:, x]
            X0 = X0[x, :]
            Kva = []
            Kve = []
        self.nonmissing = x

        if len(Kva) == 0 or len(Kve) == 0:
            if self.verbose:
                sys.stderr.write("Obtaining eigendecomposition for %dx%d matrix\n" % (K.shape[0], K.shape[1]))
            begin = time.time()
            Kva, Kve = linalg.eigh(K)
            end = time.time()
            if self.verbose:
                sys.stderr.write("Total time: %0.3f\n" % (end - begin))

        self.K = K
        self.Kva = Kva
        self.Kve = Kve
        self.N = self.K.shape[0]
        self.Y = Y.reshape((self.N, 1))
        self.X0 = X0

        if sum(self.Kva < 1e-6):
            if self.verbose: sys.stderr.write("Cleaning %d eigenvalues\n" % (sum(self.Kva < 0)))
            self.Kva[self.Kva < 1e-6] = 1e-6

        self.transform()

    def transform(self):

        """
        Computes a transformation on the phenotype vector and the covariate matrix.
        The transformation is obtained by left multiplying each parameter by the transpose of the
        eigenvector matrix of K (the kinship).
        """

        self.Yt = matrixMult(self.Kve.T, self.Y)
        self.X0t = matrixMult(self.Kve.T, self.X0)
        self.X0t_stack = np.hstack([self.X0t, np.ones((self.N, 1))])
        self.q = self.X0t.shape[1]

    def getMLSoln(self, h, X):
        """
        Obtains the maximum-likelihood estimates for the covariate coefficients (beta),
        the total variance of the trait (sigma) and also passes intermediates that can
        be utilized in other functions. The input parameter h is a value between 0 and 1 and represents
        the heritability or the proportion of the total variance attributed to genetics.  The X is the
        covariate matrix.
        """

        S = 1.0 / (h * self.Kva + (1.0 - h))
        Xt = X.T * S
        XX = matrixMult(Xt, X)
        try:
            XX_i = linalg.inv(XX)
        except numpy.linalg.linalg.LinAlgError:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        beta = matrixMult(matrixMult(XX_i, Xt), self.Yt)
        Yt = self.Yt - matrixMult(X, beta)
        Q = np.dot(Yt.T * S, Yt)
        sigma = Q * 1.0 / (float(self.N) - float(X.shape[1]))
        return beta, sigma, Q, XX_i, XX

    def LL_brent(self, h, X=None, REML=False):
        # brent will not be bounded by the specified bracket.
        # I return a large number if we encounter h < 0 to avoid errors in LL computation during the search.
        if h < 0: return 1e6
        return -self.LL(h, X, stack=False, REML=REML)[0]

    def LL(self, h, X=None, stack=True, REML=False):
        """
        Computes the log-likelihood for a given heritability (h).  If X==None, then the
        default X0t will be used.  If X is set and stack=True, then X0t will be matrix concatenated with
        the input X.  If stack is false, then X is used in place of X0t in the LL calculation.
        REML is computed by adding additional terms to the standard LL and can be computed by setting REML=True.
        """

        if X is None:
            X = self.X0t
        elif stack:
            self.X0t_stack[:, (self.q)] = matrixMult(self.Kve.T, X)[:, 0]
            X = self.X0t_stack

        n = float(self.N)
        q = float(X.shape[1])
        beta, sigma, Q, XX_i, XX = self.getMLSoln(h, X)
        LL = n * np.log(2 * np.pi) + np.log(h * self.Kva + (1.0 - h)).sum() + n + n * np.log(1.0 / n * Q)
        LL = -0.5 * LL

        if REML:
            LL_REML_part = q * np.log(2.0 * np.pi * sigma) + np.log(linalg.det(matrixMult(X.T, X))) - np.log(
                linalg.det(XX))
            LL = LL + 0.5 * LL_REML_part

        LL = LL.sum()
        return LL, beta, sigma, XX_i

    def getMax(self, H, X=None, REML=False):
        """
        Helper functions for .fit(...).
        This function takes a set of LLs computed over a grid and finds possible regions
        containing a maximum.  Within these regions, a Brent search is performed to find the
        optimum.
        """

        n = len(self.LLs)
        HOpt = []
        for i in range(1, n - 2):
            if self.LLs[i - 1] < self.LLs[i] and self.LLs[i] > self.LLs[i + 1]:
                HOpt.append(optimize.brent(self.LL_brent, args=(X, REML), brack=(H[i - 1], H[i + 1])))
                if np.isnan(HOpt[-1]): HOpt[-1] = H[i - 1]
                #if np.isnan(HOpt[-1]): HOpt[-1] = self.LLs[i-1]
                #if np.isnan(HOpt[-1][0]): HOpt[-1][0] = [self.LLs[i-1]]

        if len(HOpt) > 1:
            if self.verbose: sys.stderr.write("NOTE: Found multiple optima.  Returning first...\n")
            return HOpt[0]
        elif len(HOpt) == 1:
            return HOpt[0]
        elif self.LLs[0] > self.LLs[n - 1]:
            return H[0]
        else:
            return H[n - 1]

    def fit(self, X=None, ngrids=100, REML=True):
        """
        Finds the maximum-likelihood solution for the heritability (h) given the current parameters.
        X can be passed and will transformed and concatenated to X0t.  Otherwise, X0t is used as
        the covariate matrix.

        This function calculates the LLs over a grid and then uses .getMax(...) to find the optimum.
        Given this optimum, the function computes the LL and associated ML solutions.
        """

        if X is None:
            X = self.X0t
        else:
            # X = np.hstack([self.X0t,matrixMult(self.Kve.T, X)])
            self.X0t_stack[:, (self.q)] = matrixMult(self.Kve.T, X)[:, 0]
            X = self.X0t_stack

        H = np.array(range(ngrids)) / float(ngrids)
        L = np.array([self.LL(h, X, stack=False, REML=REML)[0] for h in H])
        self.LLs = L

        hmax = self.getMax(H, X, REML)
        L, beta, sigma, betaSTDERR = self.LL(hmax, X, stack=False, REML=REML)

        self.H = H
        self.optH = hmax.sum()
        self.optLL = L
        self.optBeta = beta
        self.optSigma = sigma.sum()

        return hmax, beta, sigma, L

    def association(self, X, h=None, stack=True, REML=True, returnBeta=False):
        """
        Calculates association statitics for the SNPs encoded in the vector X of size n.
        If h == None, the optimal h stored in optH is used.
        """

        if stack:
            #X = np.hstack([self.X0t,matrixMult(self.Kve.T, X)])
            self.X0t_stack[:, (self.q)] = matrixMult(self.Kve.T, X)[:, 0]
            X = self.X0t_stack

        if h == None: h = self.optH

        L, beta, sigma, betaVAR = self.LL(h, X, stack=False, REML=REML)
        q = len(beta)
        ts, ps = self.tstat(beta[q - 1], betaVAR[q - 1, q - 1], sigma, q)

        if returnBeta: return ts, ps, beta[q - 1].sum(), betaVAR[q - 1, q - 1].sum() * sigma
        return ts, ps

    def tstat(self, beta, var, sigma, q, log=False):
        """
        Calculates a t-statistic and associated p-value given the estimate of beta and its standard error.
        This is actually an F-test, but when only one hypothesis is being performed, it reduces to a t-test.
        """

        ts = beta / np.sqrt(var * sigma)
        #ps = 2.0*(1.0 - stats.t.cdf(np.abs(ts), self.N-q))
        # sf == survival function - this is more accurate -- could also use logsf if the precision is not good enough
        if log:
            ps = 2.0 + (stats.t.logsf(np.abs(ts), self.N - q))
        else:
            ps = 2.0 * (stats.t.sf(np.abs(ts), self.N - q))
        if not len(ts) == 1 or not len(ps) == 1:
            raise Exception("Something bad happened :(")
        return ts.sum(), ps.sum()

    def plotFit(self, color='b-', title=''):
        """
        Simple function to visualize the likelihood space.  It takes the LLs
        calcualted over a grid and normalizes them by subtracting off the mean and exponentiating.
        The resulting "probabilities" are normalized to one and plotted against heritability.
        This can be seen as an approximation to the posterior distribuiton of heritability.

        For diagnostic purposes this lets you see if there is one distinct maximum or multiple
        and what the variance of the parameter looks like.
        """

        import matplotlib.pyplot as pl

        mx = self.LLs.max()
        p = np.exp(self.LLs - mx)
        p = p / p.sum()

        pl.plot(self.H, p, color)
        pl.xlabel("Heritability")
        pl.ylabel("Probability of data")
        pl.title(title)

    def meanAndVar(self):

        mx = self.LLs.max()
        p = np.exp(self.LLs - mx)
        p = p / p.sum()

        mn = (self.H * p).sum()
        vx = ((self.H - mn) ** 2 * p).sum()

        return mn, vx


class LMM_withK2:
    """
        This is a simple version of EMMA/fastLMM.
        The main purpose of this module is to take a phenotype vector (Y), a set of covariates (X)
        and 2 kinship matrices (K, K2)
        and to optimize this model by finding the maximum-likelihood estimates for the model parameters.
        There are three model parameters: heritability (h), covariate coefficients (beta) and the total
        phenotypic variance (sigma).
        Heritability as defined here is the proportion of the total variance (sigma) that is attributed to
        the kinship matrix.
        For simplicity, we assume that everything being input is a numpy array.
        If this is not the case, the module may throw an error as conversion from list to numpy array
        is not done consistently.
    """

    def __init__(self, Y, K, Kva=[], Kve=[], X0=None, verbose=False, K2=None, K2va=[], K2ve=[]):
        """
        The constructor takes a phenotype vector or array of size n.
        It takes a kinship matrix of size n x n.  Kva and Kve can be computed as Kva,Kve = linalg.eigh(K) and cached.
        If they are not provided, the constructor will calculate them.
        X0 is an optional covariate matrix of size n x q, where there are q covariates.
        When this parameter is not provided, the constructor
        will set X0 to an n x 1 matrix of all ones to represent a mean effect.
        """
        if X0 is None:
            X0 = np.ones(len(Y)).reshape(len(Y), 1)
        self.verbose = verbose

        x = True - np.isnan(Y)
        x = x.reshape(-1, )
        x, Y, X0, K, Kva, Kve, K2, K2va, K2ve = \
            remove_missing_values(x, Y, X0, self.verbose, K, Kva, Kve, K2, K2va, K2ve)

        self.nonmissing = x

        if len(Kva) == 0 or len(Kve) == 0:
            do_eigendecomposition(K, 'K1', self.verbose)

        if len(K2va) == 0 or len(K2ve) == 0:
            do_eigendecomposition(K, 'K2', self.verbose)

        if sum(Kva < 1e-6):
            clean_eigenvalues(Kva, self.verbose)

        if sum(K2va < 1e-6):
            clean_eigenvalues(K2va, self.verbose)


        self.N = self.K.shape[0]
        self.Y = Y.reshape((self.N, 1))
        self.X0 = X0
        self.K = K
        self.Kva = Kva
        self.Kve = Kve
        self.K2 = K2
        self.K2va = K2va
        self.K2ve = K2ve

        self.transform()

    def transform(self):

        """
        Computes a transformation on the phenotype vector and the covariate matrix.
        The transformation is obtained by left multiplying each parameter by the transpose of the
        eigenvector matrix of K (the kinship).
        """

        self.Yt = matrixMult(self.Kve.T, self.Y)
        self.X0t = matrixMult(self.Kve.T, self.X0)
        self.X0t_stack = np.hstack([self.X0t, np.ones((self.N, 1))])
        self.q = self.X0t.shape[1]

    def getMLSoln(self, h, X):
        """
        Obtains the maximum-likelihood estimates for the covariate coefficients (beta),
        the total variance of the trait (sigma) and also passes intermediates that can
        be utilized in other functions. The input parameter h is a value between 0 and 1 and represents
        the heritability or the proportion of the total variance attributed to genetics.  The X is the
        covariate matrix.
        """

        S = 1.0 / (h * self.Kva + (1.0 - h))
        Xt = X.T * S
        XX = matrixMult(Xt, X)
        try:
            XX_i = linalg.inv(XX)
        except numpy.linalg.linalg.LinAlgError:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        beta = matrixMult(matrixMult(XX_i, Xt), self.Yt)
        Yt = self.Yt - matrixMult(X, beta)
        Q = np.dot(Yt.T * S, Yt)
        sigma = Q * 1.0 / (float(self.N) - float(X.shape[1]))
        return beta, sigma, Q, XX_i, XX

    def LL_brent(self, h, X=None, REML=False):
        # brent will not be bounded by the specified bracket.
        # I return a large number if we encounter h < 0 to avoid errors in LL computation during the search.
        if h < 0: return 1e6
        return -self.LL(h, X, stack=False, REML=REML)[0]

    def LL(self, h, X=None, stack=True, REML=False):
        """
        Computes the log-likelihood for a given heritability (h).  If X==None, then the
        default X0t will be used.  If X is set and stack=True, then X0t will be matrix concatenated with
        the input X.  If stack is false, then X is used in place of X0t in the LL calculation.
        REML is computed by adding additional terms to the standard LL and can be computed by setting REML=True.
        """

        if X is None:
            X = self.X0t
        elif stack:
            self.X0t_stack[:, (self.q)] = matrixMult(self.Kve.T, X)[:, 0]
            X = self.X0t_stack

        n = float(self.N)
        q = float(X.shape[1])
        beta, sigma, Q, XX_i, XX = self.getMLSoln(h, X)
        LL = n * np.log(2 * np.pi) + np.log(h * self.Kva + (1.0 - h)).sum() + n + n * np.log(1.0 / n * Q)
        LL = -0.5 * LL

        if REML:
            LL_REML_part = q * np.log(2.0 * np.pi * sigma) + np.log(linalg.det(matrixMult(X.T, X))) - np.log(
                linalg.det(XX))
            LL = LL + 0.5 * LL_REML_part

        LL = LL.sum()
        return LL, beta, sigma, XX_i

    def getMax(self, H, X=None, REML=False):
        """
        Helper functions for .fit(...).
        This function takes a set of LLs computed over a grid and finds possible regions
        containing a maximum.  Within these regions, a Brent search is performed to find the
        optimum.
        """

        n = len(self.LLs)
        HOpt = []
        for i in range(1, n - 2):
            if self.LLs[i - 1] < self.LLs[i] and self.LLs[i] > self.LLs[i + 1]:
                HOpt.append(optimize.brent(self.LL_brent, args=(X, REML), brack=(H[i - 1], H[i + 1])))
                if np.isnan(HOpt[-1]): HOpt[-1] = H[i - 1]
                #if np.isnan(HOpt[-1]): HOpt[-1] = self.LLs[i-1]
                #if np.isnan(HOpt[-1][0]): HOpt[-1][0] = [self.LLs[i-1]]

        if len(HOpt) > 1:
            if self.verbose: sys.stderr.write("NOTE: Found multiple optima.  Returning first...\n")
            return HOpt[0]
        elif len(HOpt) == 1:
            return HOpt[0]
        elif self.LLs[0] > self.LLs[n - 1]:
            return H[0]
        else:
            return H[n - 1]

    def fit(self, X=None, ngrids=100, REML=True):
        """
        Finds the maximum-likelihood solution for the heritability (h) given the current parameters.
        X can be passed and will transformed and concatenated to X0t.  Otherwise, X0t is used as
        the covariate matrix.

        This function calculates the LLs over a grid and then uses .getMax(...) to find the optimum.
        Given this optimum, the function computes the LL and associated ML solutions.
        """

        if X is None:
            X = self.X0t
        else:
            # X = np.hstack([self.X0t,matrixMult(self.Kve.T, X)])
            self.X0t_stack[:, (self.q)] = matrixMult(self.Kve.T, X)[:, 0]
            X = self.X0t_stack

        H = np.array(range(ngrids)) / float(ngrids)
        L = np.array([self.LL(h, X, stack=False, REML=REML)[0] for h in H])
        self.LLs = L

        hmax = self.getMax(H, X, REML)
        L, beta, sigma, betaSTDERR = self.LL(hmax, X, stack=False, REML=REML)

        self.H = H
        self.optH = hmax.sum()
        self.optLL = L
        self.optBeta = beta
        self.optSigma = sigma.sum()

        return hmax, beta, sigma, L

    def association(self, X, h=None, stack=True, REML=True, returnBeta=False):
        """
        Calculates association statitics for the SNPs encoded in the vector X of size n.
        If h == None, the optimal h stored in optH is used.
        """

        if stack:
            #X = np.hstack([self.X0t,matrixMult(self.Kve.T, X)])
            self.X0t_stack[:, (self.q)] = matrixMult(self.Kve.T, X)[:, 0]
            X = self.X0t_stack

        if h == None: h = self.optH

        L, beta, sigma, betaVAR = self.LL(h, X, stack=False, REML=REML)
        q = len(beta)
        ts, ps = self.tstat(beta[q - 1], betaVAR[q - 1, q - 1], sigma, q)

        if returnBeta: return ts, ps, beta[q - 1].sum(), betaVAR[q - 1, q - 1].sum() * sigma
        return ts, ps

    def tstat(self, beta, var, sigma, q, log=False):
        """
        Calculates a t-statistic and associated p-value given the estimate of beta and its standard error.
        This is actually an F-test, but when only one hypothesis is being performed, it reduces to a t-test.
        """

        ts = beta / np.sqrt(var * sigma)
        #ps = 2.0*(1.0 - stats.t.cdf(np.abs(ts), self.N-q))
        # sf == survival function - this is more accurate -- could also use logsf if the precision is not good enough
        if log:
            ps = 2.0 + (stats.t.logsf(np.abs(ts), self.N - q))
        else:
            ps = 2.0 * (stats.t.sf(np.abs(ts), self.N - q))
        if not len(ts) == 1 or not len(ps) == 1:
            raise Exception("Something bad happened :(")
        return ts.sum(), ps.sum()

    def plotFit(self, color='b-', title=''):
        """
        Simple function to visualize the likelihood space.  It takes the LLs
        calcualted over a grid and normalizes them by subtracting off the mean and exponentiating.
        The resulting "probabilities" are normalized to one and plotted against heritability.
        This can be seen as an approximation to the posterior distribuiton of heritability.

        For diagnostic purposes this lets you see if there is one distinct maximum or multiple
        and what the variance of the parameter looks like.
        """

        import matplotlib.pyplot as pl

        mx = self.LLs.max()
        p = np.exp(self.LLs - mx)
        p = p / p.sum()

        pl.plot(self.H, p, color)
        pl.xlabel("Heritability")
        pl.ylabel("Probability of data")
        pl.title(title)

    def meanAndVar(self):

        mx = self.LLs.max()
        p = np.exp(self.LLs - mx)
        p = p / p.sum()

        mn = (self.H * p).sum()
        vx = ((self.H - mn) ** 2 * p).sum()

        return mn, vx

