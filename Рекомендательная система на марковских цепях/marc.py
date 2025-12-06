#!/usr/bin/env python3
# numeric_mdp_example.py
"""
Policy Iteration for the 3-state numeric example
from the LaTeX demonstration.

States: s1, s2, s3
Actions: a1, a2
"""

import numpy as np

# -----------------------------------------
# POLICY EVALUATION (exact)
# -----------------------------------------

def policy_evaluation_exact(policy, tr, R, gamma=0.9):
    S = tr.shape[0]
    Ppi = np.zeros((S, S))
    Rpi = np.zeros(S)

    for s in range(S):
        a = policy[s]
        Ppi[s] = tr[s, a]
        Rpi[s] = R[s, a]

    A = np.eye(S) - gamma * Ppi
    V = np.linalg.solve(A, Rpi)
    return V


# -----------------------------------------
# POLICY ITERATION
# -----------------------------------------

def policy_iteration_exact(tr, R, gamma=0.9, max_iter=50):
    S, A, _ = tr.shape
    # Initial greedy policy by immediate reward
    policy = np.argmax(R, axis=1).astype(int)
    stable = False

    while not stable:
        V = policy_evaluation_exact(policy, tr, R, gamma)
        stable = True
        for s in range(S):
            Q = R[s] + gamma * (tr[s] @ V)
            best = np.argmax(Q)
            if best != policy[s]:
                policy[s] = best
                stable = False

    return policy, V  # fallback


# -----------------------------------------
# MAIN: numeric example from LaTeX
# -----------------------------------------

def main():
    # States: s1, s2, s3
    # Actions: a1=0, a2=1

    gamma = 0.9

    # Reward matrix R(s,a)
    R = np.array([
        [1, 1],   # s1
        [1, 1],   # s2
        [2, 0]    # s3
    ], dtype=float)

    # Transition probabilities tr[s,a,s']
    tr = np.zeros((3, 2, 3))

    # s1 transitions:
    tr[0, 0] = [0.5, 0.5, 0.0]   # a1
    tr[0, 1] = [0.4, 0.6, 0.0]   # a2

    # s2 transitions:
    tr[1, 0] = [0.0, 1.0, 0.0]   # a1
    tr[1, 1] = [0.0, 0.0, 1.0]   # a2

    # s3 transitions:
    tr[2, 0] = [1.0, 0.0, 0.0]   # a1
    tr[2, 1] = [0.0, 0.0, 1.0]   # a2

    print("\n=== Numeric MDP Example ===")
    print("Reward matrix R:")
    print(R)
    print("\nTransition tensor tr[s,a,s']:")
    print(tr)

    # Run policy iteration
    policy, V = policy_iteration_exact(tr, R, gamma=gamma)

    # Output
    actions = ["a1", "a2"]
    states = ["s1", "s2", "s3"]

    print("\nOptimal policy:")
    for i in range(3):
        print(f"  {states[i]} -> {actions[policy[i]]}")

    print("\nOptimal value function V*:")
    for i in range(3):
        print(f"  V({states[i]}) = {V[i]:.4f}")


if __name__ == "__main__":
    main()
