# This script sets up an online logistic regression model (FTLR).

import numpy as np
from math import exp, sqrt, log

def log_loss(y, p):
    """
    A function to compute the log loss of a predicted probability p given
    a true target y.
    :param y: True target value
    :param p: Predicted probability
    :return: Log loss.
    """
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1 else -log(1. - p)


class FTRLP():
    """
    --- Follow The Regularized Leader - Proximal ---
    FTRL-P is an online classification algorithm that combines both L1 and L2
    norms, particularly suited for large data sets with extremely high dimensionality.
    This implementation follow the algorithm by H. B. McMahan et. al. It minimizes
    the LogLoss function iteratively with a combination of L2 and L1 (centralized
    at the current point) norms and adaptive, per coordinate learning rates.
    This algorithm is efficient at obtaining sparsity and has proven to perform
    very well in massive Click-Through-Rate prediction tasks.
    References:
        * Follow-the-Regularized-Leader and Mirror Descent: Equivalent Theorems
          and L1 Regularization, H. Brendan McMahan
        * Ad Click Prediction: a View from the Trenches, H. Brendan McMahan et. al.
    """
    
    def __init__(
            self,
            alpha=0.1,
            beta=1.0,
            l1=1.0,
            l2=1.0,
            D=2**20,
            encoder=None,
            name=None,
            n=None,
            z=None,
            subsample=1,
            epochs=1,
            rate=0,
            w=1.0,
    ):
        """ The per feature learning rate is given by:
              eta = alpha / ( beta + sqrt( sum g**g ) )
        """
        self.alpha, self.beta, self.l1, self.l2, self.D, self.encoder = alpha, beta, l1, l2, D, encoder
        self.log_likelihood_, self.loss = 0, []
        self.name = name
        self.w = w 

        self.z, self.n = [0.] * D, [0.] * D

        self.coef_, self.cname = {}, None
        self.subsample, self.rate, self.epochs = subsample, rate, epochs
        self.target_ratio = 0.0   

    def clear_params(self):
        self.log_likelihood_, self.loss, self.coef_, self.cname = 0, [], {}, None
        self.z, self.n = [0.] * self.D, [0.] * self.D

    def _update(self, y, p, x, w):
        """
        # --- Update weight vector and learning rate.
        # With the prediction round completed we can proceed to
        # updating the weight vector z and the learning rate eta
        # based on the last observed label.
        # To do so we will use the computed probability and target
        # value to find the gradient loss and continue from there.
        # The gradient for the log likelihood for round t can easily
        # be shown to be:
        #               g_i = (p - y) * x_i, (round t)
        # The remaining quantities are updated according to the
        # minimization procedure outlined in [2].
        :param y: True target variable
        :param p: Predicted probability for the current sample
        :param x: Non zero feature values
        :param w: Weights
        :return: Nothing
        """
        for i in x.keys(): # x is feature vector
            g = (p-y) * x[i]

            # --- Update constant sigma
            # Note that this upgrade is equivalent to
            #       (eta_(t, i))^-1 - (eta_(t - 1, i))^-1
            # as discussed in [2].
            s = (sqrt(self.n[i] + g * g) - sqrt(self.n[i])) / self.alpha
            self.z[i] += g - s * w[i]
            self.n[i] += g * g
    
    def _dot(self, x):
        """ takes the feature vector and dots it into the weights"""
        wtx, w = 0, {}
        for indx in x.keys():
            # print indx, len(self.z), self.D
            if abs(self.z[indx]) <= self.l1:
                w[indx] = 0
            else:
                sign = np.sign(self.z[indx])
                w[indx] = - (self.z[indx] - sign * self.l1) / \
                            (self.l2 + (self.beta + sqrt(self.n[indx])) / self.alpha)            
            wtx += w[indx] * x[indx]
        return wtx, w

    def _dump_weights(self):
        weights = {}
        for z_ind, z_value in enumerate(self.z):
            if abs(z_value) < self.l1:
                weights[z_ind] = 0.0
            else:
                weights[z_ind] = - (z_value - np.sign(z_value) * self.l1) / \
                            (self.l2 + (self.beta + sqrt(self.n[z_ind])) / self.alpha)
        return weights

    def _train_one(self, t, x, y):
        self.target_ratio = (1.0 * (t * self.target_ratio + y)) / (t + 1)
        wtx, w = self._dot(x)
        p = 1. / (1. + exp(-max(min(wtx, 35.), -35.)))
        self.log_likelihood_ += log_loss(y, p)
        self._update(y, p, x, w)
        return p
    
    def encode(self, X):
        return self.encoder.encode(X)

    def fit(self, Xs, Ys):
        for i, v in enumerate(Xs):
            self._train_one(0, v, Ys[i])

    def predict_one(self, x):
        wtx, w = self._dot(x)
        p = 1. / (1. + exp(-max(min(wtx, 35.), -35.)))
        q = p / (p + (1-p) / self.w)
        return (1-q, q)

    def predict_proba(self, xs):
        result = []
        for x in xs:
            wtx, w = self._dot(x)
            p = 1. / (1. + exp(-max(min(wtx, 35.), -35.)))
            q = p / (p + (1-p) / self.w)
            result.append((1-q, q))
        return np.array(result)
