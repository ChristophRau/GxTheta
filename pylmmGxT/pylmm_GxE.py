from lmm import *
import numpy as np
from scipy import linalg

def GWAS_GxE(Y, snp, exposure, args={}):
    """
    :param Y:        n x 1 output
    :param snp:      n x m array of snps
    :param exposure: n x 1 covariate that has a (small) number of levels.
                        kinship is 0 for individuals that do not have the same
                        exposure value
    :param args:     any other args to pass on to the GWAS function (see lmm.py)
    :return:         Ts, Ps (list of T-statistics, list of p-values, respectively)
    """
    exposure_levels = set(exposure)
    sorted_snps = []
    sorted_Ks = []
    sorted_exposures = []
    sorted_Ys = []
    for level in exposure_levels:
        # We need to calculate the kinship separately for each value of
        # the exposure. Sorting the data will make it "look" nicer
        # if we ever want to print it out as well.
        mask = exposure == level
        sorted_exposure = exposure[mask]
        sorted_snp = snp[mask]
        sorted_K = calculateKinship(sorted_snp)
        sorted_Y = Y[mask]

        sorted_exposures.append(sorted_exposure)
        sorted_snps.append(sorted_snp)
        sorted_Ks.append(sorted_K)
        sorted_Ys.append(sorted_Y)

    snps_GxE = np.vstack(sorted_snps)
    K_GxE = linalg.block_diag(*sorted_Ks)
    # Kinship is 0 between individuals with different
    # levels for their exposures, so the matrix has
    # a block-diagonal structure.
    Kva_GxE, Kve_GxE = linalg.eigh(K_GxE)
    exposures_GxE = np.concatenate(sorted_exposures)
    Y_GxE = np.concatenate(sorted_Ys)

    all_outputs = []
    for snp_GxE in snps_GxE.T:
        # We are testing the significance of the GxE effect
        # after regressing out the effect of the environment
        # and the effect of the gene for which we're looking for a GxE effect
        snp_exposure = snp_GxE * exposures_GxE
        snp_exposure = snp_exposure.reshape(len(snp_exposure), -1)
        X0_GxE = np.vstack((np.ones(len(snp_GxE)), snp_GxE, exposures_GxE)).T
        output = GWAS(Y_GxE, snp_exposure, K_GxE, Kva=Kva_GxE, Kve=Kve_GxE, X0=X0_GxE, **args)
        T, P = output[0][0], output[1][0]
        all_outputs.append((T, P))
    Ts = [x[0] for x in all_outputs]
    Ps = [x[1] for x in all_outputs]
    return Ts, Ps

if __name__ == "__main__":
    study_size = 1000
    n_rand_genes = 268
    ## Pick some random data
    rand_gene_mafs = np.random.uniform(0.1, 0.5, size=n_rand_genes)
    rand_genes = [np.random.binomial(2, rand_gene_mafs[i], study_size)
                  for i in range(len(rand_gene_mafs))]
    rand_genes = np.array(rand_genes)
    # print rand_genes.shape
    # print rand_genes
    ## The rand_genes should have no population structure
    exposure_values = np.array([0]*(study_size/2) + [1]*(study_size/2))
    np.random.shuffle(exposure_values)
    ## Shuffle the exposures to make sure the un-shuffling works properly
    GxE_maf = .4
    GxE_gene = np.random.binomial(2, GxE_maf, study_size)
    Y = GxE_gene*exposure_values + np.random.normal(size=study_size)
    genes = np.vstack((GxE_gene, rand_genes))
    ## The first output should have a very strong t-statistic and p-value
    ## The rest should all be very insignificant.
    genes = genes.T
    # print genes.shape
    Ts, Ps = GWAS_GxE(Y=Y, snp=genes, exposure=exposure_values)
    for i in range(20):
        if i == 0:
            print 'T-statistic\tp-value'
        print Ts[i], Ps[i]
    print 'First row should have very high t-statistic/significant p-value'