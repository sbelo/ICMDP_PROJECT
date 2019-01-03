from ortools.linear_solver import pywraplp
import numpy as np
from tqdm import tqdm


# Solves inverse cdmp problem with constraint generation
# This is the "heavy" or full version
# Input: list of contexts, list of experts feature expectations, list of weights, function which computes
#        feature expectations for a specific reward coefficient vector w.
#        for contexts[i], estimation of experts feature expectations are expert_feature_expectations[i]  which
#        are based on weights[i] observed trajectories.

def constraint_generation_solver(contexts: list, expert_feature_expectations: list, weights: list,
                                 feature_expectations):
    assert (len(contexts) == len(expert_feature_expectations) == len(weights))
    num_features = len(expert_feature_expectations[0])
    context_dim = len(contexts[0])

    # Initialize "random" policies PI^c(i)
    W = np.ones((context_dim, num_features))
    W = W / W.sum()
    PI = {}
    for context in contexts:
        PI[tuple(context)] = [(feature_expectations(context @ W)).M]

    num_perm = pow(2,num_features*context_dim)
    const_coef = np.ones([num_perm,num_features*context_dim])
    for i in tqdm(range(num_perm)):
        for j in range(num_features*context_dim):
            if i == 0:
                continue
            else:
                if j == 0:
                    const_coef[i,j] = - const_coef[i-1,j]
                elif const_coef[i,j-1] > 0 and const_coef[i-1,j-1] < 0:
                    const_coef[i,j] = - const_coef[i-1,j]
                else:
                    const_coef[i,j] = const_coef[i-1,j]
    it = 0

    # Do some number of times
    while True:
        it += 1
        print(it)
        solver = pywraplp.Solver('LinearExample', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

        # Initialize W variable
        Wvar = [[]] * context_dim
        for i in range(context_dim):
            for j in range(num_features):
                Wvar[i].append(solver.NumVar(-1.0, 1.0, "W_" + str(i) + "_" + str(j)))

        # Initialize Objective
        objective = solver.Objective()
        u = []
        for i in range(len(expert_feature_expectations)):
            u_PI = np.zeros(num_features)
            for u_pi in PI[tuple(contexts[i])]:
                u_PI += u_pi
            u_PI /= len(PI[tuple(contexts[i])])
            u.append(expert_feature_expectations[i] - u_PI)
        # At this point the objective function is sum over i: weights[i] * (context[i] @ W @ u[i])
        for i in range(context_dim):
            for j in range(num_features):
                coeff = 0
                dummy = np.zeros((context_dim, num_features))
                dummy[i, j] = 1.0
                for index in range(len(contexts)):
                    coeff += weights[index] * (contexts[index] @ dummy @ u[i])
                objective.SetCoefficient(Wvar[i][j], coeff)

        # Initialize constraint
        objective.SetMaximization()
        constraint = []
        for k in range(num_perm):
            constraint.append(solver.Constraint(-1.0, 1.0))
            for i in range(context_dim):
                for j in range(num_features):
                    constraint[k].SetCoefficient(Wvar[i][j], const_coef[k,num_features*i + j])

        # Solve
        status = solver.Solve()
        if status != solver.OPTIMAL:
            if status == solver.FEASIBLE:
                print("A potentially suboptimal solution was found.")
            else:
                print("The solver could not solve the problem.")

        # Build numpy matrix of solution
        W = np.zeros((context_dim, num_features))
        for i in range(context_dim):
            for j in range(num_features):
                W[i, j] = Wvar[i][j].solution_value()

        # Stop condition TODO: come up with something better
        # print(W)
        if it >= 20:
            return W

        # Update PI^c(i)'s:
        for context in contexts:
            PI[tuple(context)].append((feature_expectations(context @ W)).M)