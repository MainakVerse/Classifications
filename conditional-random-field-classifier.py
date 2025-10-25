import numpy as np

class ConditionalRandomFieldScratch:
    def __init__(self, n_states, n_features, learning_rate=0.01, n_iters=100):
        """
        n_states: number of possible output labels
        n_features: number of input features per timestep
        """
        self.n_states = n_states
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.n_iters = n_iters

        # Model parameters
        self.W = np.random.randn(n_states, n_features) * 0.01  # Feature weights
        self.T = np.random.randn(n_states, n_states) * 0.01    # Transition weights

    def _score(self, X, y):
        """Compute total sequence score"""
        score = np.sum(self.W[y, :] * X)
        score += np.sum(self.T[y[:-1], y[1:]])
        return score

    def _forward(self, X):
        """Forward algorithm (log-space)"""
        n_steps = X.shape[0]
        alpha = np.full((n_steps, self.n_states), -np.inf)
        alpha[0] = np.dot(self.W, X[0])

        for t in range(1, n_steps):
            for j in range(self.n_states):
                alpha[t, j] = np.logaddexp.reduce(alpha[t - 1] + self.T[:, j]) + np.dot(self.W[j], X[t])
        return alpha

    def _backward(self, X):
        """Backward algorithm (log-space)"""
        n_steps = X.shape[0]
        beta = np.full((n_steps, self.n_states), -np.inf)
        beta[-1] = 0

        for t in range(n_steps - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.logaddexp.reduce(self.T[i, :] + np.dot(self.W, X[t + 1]) + beta[t + 1])
        return beta

    def _log_partition(self, X):
        """Compute log partition function"""
        alpha = self._forward(X)
        return np.logaddexp.reduce(alpha[-1])

    def fit(self, X_list, y_list):
        """Train CRF via gradient ascent on log-likelihood"""
        for _ in range(self.n_iters):
            dW = np.zeros_like(self.W)
            dT = np.zeros_like(self.T)

            for X, y in zip(X_list, y_list):
                alpha = self._forward(X)
                beta = self._backward(X)
                logZ = np.logaddexp.reduce(alpha[-1])

                # Empirical feature counts
                for t in range(len(y)):
                    dW[y[t]] += X[t]
                    if t > 0:
                        dT[y[t - 1], y[t]] += 1

                # Expected feature counts
                for t in range(len(y)):
                    marginals = np.exp(alpha[t] + beta[t] - logZ)
                    dW -= marginals[:, None] * X[t]
                    if t > 0:
                        pairwise = np.exp(
                            np.add.outer(alpha[t - 1], np.dot(self.W, X[t])) +
                            self.T + beta[t] - logZ
                        )
                        dT -= pairwise / np.sum(pairwise)

            # Gradient update
            self.W += self.learning_rate * dW
            self.T += self.learning_rate * dT

    def predict(self, X):
        """Viterbi decoding for most likely label sequence"""
        n_steps = X.shape[0]
        delta = np.zeros((n_steps, self.n_states))
        psi = np.zeros((n_steps, self.n_states), dtype=int)

        delta[0] = np.dot(self.W, X[0])
        for t in range(1, n_steps):
            for j in range(self.n_states):
                seq_scores = delta[t - 1] + self.T[:, j]
                psi[t, j] = np.argmax(seq_scores)
                delta[t, j] = np.max(seq_scores) + np.dot(self.W[j], X[t])

        y_pred = np.zeros(n_steps, dtype=int)
        y_pred[-1] = np.argmax(delta[-1])
        for t in range(n_steps - 2, -1, -1):
            y_pred[t] = psi[t + 1, y_pred[t + 1]]

        return y_pred
