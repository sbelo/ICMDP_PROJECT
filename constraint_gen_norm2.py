from qpsolvers import solve_qp
import numpy as np


# Solves inverse cdmp problem with constraint generation
# This is the "heavy" or full version
# Input: list of contexts, list of experts feature expectations, list of weights, function which computes
#        feature expectations for a specific reward coefficient vector w.
#        for contexts[i], estimation of experts feature expectations are expert_feature_expectations[i]  which
#        are based on weights[i] observed trajectories.

def constraint_generation_solver_3(contexts: list, expert_feature_expectations: list, weights: list,
                                   feature_expectations):
    assert (len(contexts) == len(expert_feature_expectations) == len(weights))
    num_features = len(expert_feature_expectations[0])
    context_dim = len(contexts[0])
    w_flat_len = num_features * context_dim

    # Initialize "random" policies PI^c(i)
    W = np.ones((context_dim, num_features))
    W = W / W.sum()
    PI = {}
    for context in contexts:
        PI[tuple(context)] = [feature_expectations(context @ W).M]

    it = 0

    # Do some number of times
    while True:
        it += 1

        # Minimze (1/2)xPx + qx
        # s.t Gx <= h
        num_constraints = len(contexts) * it
        q = np.zeros(w_flat_len)
        P = np.identity(w_flat_len)
        h = -np.ones(num_constraints)
        G = np.zeros((num_constraints, w_flat_len))
        # Construct G.
        # c.T @ W @ u  =  (c @ u.T).flatten() @ W.flatten()    (W.flatten() is x)
        print(num_constraints)
        print(w_flat_len)
        i = 0
        for j in range(len(contexts)):
            for feature_expectations_1 in PI[tuple(contexts[j])]:
                G[i] = -np.outer(contexts[j], expert_feature_expectations[j] - feature_expectations_1).flatten()
                i += 1

        # Solve QP, reconstruct W
        W = (solve_qp(P, q, G, h).reshape((context_dim, num_features)))

        # Stop condition TODO: come up with something better
        if it >= 20:
            return W/ np.linalg.norm(W.flatten(), 2)

        # Update PI^c(i)'s:
        for context in contexts:
            PI[tuple(context)].append(feature_expectations(context @ W).M)