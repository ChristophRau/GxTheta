import numpy as np
import numpy.linalg as npl
import itertools

MIN_HERITABILITY_FRAC = 10**-6


def simplex(dimensions, fineness, sigma_2_p):
    possibles = itertools.product(*[np.linspace(0, sigma_2_p, fineness+1)]*dimensions)
    goods = (x for x in possibles if np.isclose(sum(x), sigma_2_p))
    betters = ([u if u != 0 else 10**-6*sigma_2_p for u in x] for x in goods)
    return betters


def log_likelihood_without_constant(Y, X, V):
    """
    Computes the log likelihood of the model according
    to the formula in from Yang et al. AJHG 2011
    "GCTA: A Tool for Genome-wide Complex Trait Analysis"
    Log Likelihood L:
    L = -1/2(log |V| + log|X'V^{-1}X| + y'Py)
    where P = V^{-1} - V^{-1}X (X'V^{-1}X)^{-1} X'V^{-1}
    """

    V_inv = npl.inv(V)
    rotated_X = np.dot(V_inv, X)
    square_rotated_X = np.dot(X.T, rotated_X)
    inv_square_rotated_X = npl.inv(square_rotated_X)
    P = V_inv - np.dot(rotated_X, np.dot(inv_square_rotated_X, rotated_X.T))

    det_part = np.log(abs(npl.det(V)))
    X_part = np.log(abs(npl.det(square_rotated_X)))
    y_part = np.dot(Y.T, np.dot(P, Y))

    LL = det_part + X_part + y_part
    return LL


def grid_search(K_G, K_GxE, X, Y, fineness):
    assert K_G.shape == K_GxE.shape
    assert K_G.shape[0] == len(Y) == len(X)
    I = np.eye(K_G.shape[0])
    sigma_2_p = np.var(Y)

    best_LL = None
    best_var_comps = None

    constant_LL_part = len(Y) * np.log(2 * np.pi)

    for var_comps in simplex(3, fineness, sigma_2_p):
        Var_G, Var_GxE, Var_E = var_comps
        V = Var_G*K_G + Var_GxE * K_GxE + Var_E * I
        # print var_comps,
        non_constant_LL = log_likelihood_without_constant(Y, X, V)
        # print non_constant_LL,
        LL = -.5*(constant_LL_part + non_constant_LL)
        # print LL

        if np.isnan(LL):
            continue

        if best_LL is None:
            best_LL = LL
            best_var_comps = var_comps
        else:
            if LL > best_LL:
                best_LL = LL
                best_var_comps = var_comps
            else:
                pass

    best_Var_G, best_Var_GxE, best_Var_E = best_var_comps
    best_GxE_combined = best_Var_G*K_G + best_Var_GxE * K_GxE
    var_comps_dict = {'V_G': best_Var_G, 'V_GxE': best_Var_GxE, 'V_E': best_Var_E}
    return var_comps_dict, best_GxE_combined


def impute_columns(X):
    for i in range(X.shape[1]):
        col = X[:, i]
        mean_val = np.nanmean(col)
        X[np.isnan(col), i] = mean_val
    return X


def make_K_GxE(K_G, env):
    GxE_mask = np.array([[1 if env1 == env2 else 0 for env2 in env]
                         for env1 in env])
    K_GxE = K_G*GxE_mask
    return K_GxE


def remove_fixed_effects(Y, env):
    env_vals = sorted(list(set(env)))
    mu = np.mean(Y)
    Y -= mu
    mu_0 = np.mean(Y[env == env_vals[0]])
    for env_value in env_vals:
        mu_i = np.mean(Y[env == env_value])
        Y[env == env_value] += mu_i - mu_0
    Y -= mu
    return Y


def main(Y, K_G, env, fineness=13):
    bad_Ys = np.isnan(Y)
    bad_envs = np.isnan(env)
    if bad_Ys.sum() != 0:
        raise ValueError("All phenotypes must be complete")
    if bad_envs.sum() != 0:
        raise ValueError("Environment data must be complete")
    good_Ys = ~np.isnan(Y)
    good_envs = ~np.isnan(env)
    complete_indiv_mask = good_Ys & good_envs

    good_Y = Y[complete_indiv_mask]
    good_env = env[complete_indiv_mask]
    good_K_G = K_G[:, complete_indiv_mask][complete_indiv_mask, :]
    good_K_GxE = make_K_GxE(good_K_G, good_env)

    good_X = np.column_stack((good_env, ))

    reml_Y = remove_fixed_effects(good_Y, good_env)

    components, combined_matrix = grid_search(K_G=good_K_G, K_GxE=good_K_GxE, X=good_X, Y=reml_Y, fineness=fineness)
    return components, combined_matrix


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    fn = '../data/snps.132k.clean.noX.fake.phenos'
    gxe_fn = '../data/snps.132k.clean.noX.fam'
    df = pd.read_csv(fn, sep=' ', header=None)
    gxe_df = pd.read_csv(gxe_fn, sep=' ', header=None)
    df2 = pd.merge(df, gxe_df, on=(0, 1), suffixes=('_pheno', '_env'))
    pheno = df2['2_pheno']
    envs = df2[4]
    print envs
    df2['beta_x'] = 0
    env_vals = list(set(envs))
    mu_0 = np.mean(pheno[envs == envs[0]])
    for env_val in env_vals:
        mu_i = np.mean(pheno[envs == env_val])
        df2.ix[envs == env_val, 'beta_x'] += (mu_i - mu_0)
    df2['restricted_pheno'] = df2['2_pheno'] - df2['beta_x']
    # print df2.head()
    kinship_fn = '../data/snps.132k.clean.noX.pylmm.kin'
    kinship_matrix = np.loadtxt(kinship_fn)
    # print kinship_matrix

    env_matrix = np.array([[1 if x == y else 0 for y in envs] for x in envs])
    # print env_matrix
    gxe_kinship_matrix = np.multiply(kinship_matrix, env_matrix)
    # print gxe_kinship_matrix

    comps, output_mat = main(Y=df2['2_pheno'], K_G=kinship_matrix, env=envs, fineness=10)
    print comps
    print output_mat