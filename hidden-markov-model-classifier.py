import numpy as np

class HiddenMarkovModelScratch:
    def __init__(self, n_states, n_observations, n_iters=100):
        """
        n_states: number of hidden states
        n_observations: number of possible observation symbols
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.n_iters = n_iters

        # Initialize transition, emission, and initial probabilities
        self.A = np.full((n_states, n_states), 1 / n_states)      # Transition
        self.B = np.full((n_states, n_observations), 1 / n_observations)  # Emission
        self.pi = np.full(n_states, 1 / n_states)                 # Initial

    def _forward(self, obs_seq):
        T = len(obs_seq)
        alpha = np.zeros((T, self.n_states))
        alpha[0] = self.pi * self.B[:, obs_seq[0]]

        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t - 1] * self.A[:, j]) * self.B[j, obs_seq[t]]
        return alpha

    def _backward(self, obs_seq):
        T = len(obs_seq)
        beta = np.zeros((T, self.n_states))
        beta[-1] = 1

        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i] * self.B[:, obs_seq[t + 1]] * beta[t + 1])
        return beta

    def fit(self, obs_seq):
        """Train HMM using Baum-Welch (EM algorithm)"""
        obs_seq = np.array(obs_seq, dtype=int)
        for _ in range(self.n_iters):
            alpha = self._forward(obs_seq)
            beta = self._backward(obs_seq)
            T = len(obs_seq)

            xi = np.zeros((T - 1, self.n_states, self.n_states))
            for t in range(T - 1):
                denom = np.sum(alpha[t] @ self.A * self.B[:, obs_seq[t + 1]] * beta[t + 1])
                for i in range(self.n_states):
                    numer = alpha[t, i] * self.A[i] * self.B[:, obs_seq[t + 1]] * beta[t + 1]
                    xi[t, i] = numer / (denom + 1e-10)

            gamma = np.sum(xi, axis=2)
            self.pi = gamma[0]
            self.A = np.sum(xi, axis=0) / (np.sum(gamma, axis=0, keepdims=True).T + 1e-10)

            gamma = np.vstack((gamma, np.sum(xi[-1], axis=0)))
            for k in range(self.n_observations):
                mask = obs_seq == k
                self.B[:, k] = np.sum(gamma[mask], axis=0)
            self.B /= np.sum(self.B, axis=1, keepdims=True) + 1e-10

    def predict(self, obs_seq):
        """Predict most likely hidden state sequence using Viterbi"""
        obs_seq = np.array(obs_seq, dtype=int)
        T = len(obs_seq)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        delta[0] = self.pi * self.B[:, obs_seq[0]]
        for t in range(1, T):
            for j in range(self.n_states):
                seq_probs = delta[t - 1] * self.A[:, j]
                psi[t, j] = np.argmax(seq_probs)
                delta[t, j] = np.max(seq_probs) * self.B[j, obs_seq[t]]

        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states
