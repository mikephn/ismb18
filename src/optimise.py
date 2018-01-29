# Copyright (c) 2017 Marius Tudor Morar
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Simple Bayesian optimisation procedure

Samples from a finite grid of possible values. Recommended usage is with
noiseless functions, but might work with noisy functions as well.
"""


from __future__ import print_function
from __future__ import absolute_import
from collections import OrderedDict
import itertools
import sys
from time import time
sys.dont_write_bytecode = True

import GPy
import numpy as np
from scipy.stats import norm as normaldist
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

np.set_printoptions(suppress=True)

try:
    xrange
except NameError:
    xrange = range

class GaussianProcess(object):
    def __init__(self, X, Y, noiseless=False):
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.scaler_X.fit(X)
        self.scaler_Y.fit(Y)

        self.X_scaled = self.scaler_X.transform(X)
        self.Y_scaled = self.scaler_Y.transform(Y)

        n_dim = X.shape[1]
        kernel = GPy.kern.Matern52(n_dim, ARD=True)
        self.model = GPy.models.GPRegression(self.X_scaled, self.Y_scaled,
                                             kernel=kernel)
        if noiseless:
            self.model.Gaussian_noise.variance.constrain_fixed(0)
        self.model.optimize_restarts(num_restarts=5, robust=True,
                                     verbose=False)
        if not noiseless:
            self.Y_smoothed = self.predict(X)[0]
        else:
            self.Y_smoothed = Y.copy()

    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        mean, var = self.model.predict(X_scaled)
        mean = self.scaler_Y.inverse_transform(mean)
        var = np.maximum(var, 1e-100) # To avoid numerical errors
        var *= self.scaler_Y.scale_ ** 2
        return mean, var

    def expected_improvement(self, X):
        Y_min = np.min(self.Y_smoothed)

        mean, var = self.predict(X)
        v = var - self.model.Gaussian_noise.variance
        v = np.maximum(v, 1e-100) # To avoid numerical errors
        std = np.sqrt(v)

        diff = Y_min - mean
        frac = diff / std            
        return diff * normaldist.cdf(frac) + std * normaldist.pdf(frac)


'''
# Example:
parameter_ranges = dict(
    alpha=range(5),
    batch_size=range(5),
    learning_rate=range(6),
    momentum=range(3),
)
'''

def optimise(error_function, parameter_ranges, max_iter=30, n_initial=3,
             use_time=False, random_state=None, previous_results=None):
    parameter_ranges = OrderedDict(sorted(parameter_ranges.iteritems()))
    X_all = np.array(list(itertools.product(*parameter_ranges.values())))
    X_unexplored = set(tuple(row) for row in X_all)

    # -------------------------------------------------------------------------
    # Load previous results
    if previous_results is not None:
        X_opt, Y_opt, T_eval, T_opt, failed_opt = previous_results
        assert X_opt.shape[1] == X_all.shape[1]

        cur_iter = X_opt.shape[0]
        assert max_iter > cur_iter

        for i in xrange(X_opt.shape[0]):
            X_unexplored.remove(tuple(X_opt[i, :]))

        X_opt = X_opt.copy()
        X_opt.resize((max_iter, X_all.shape[1]))
        Y_opt = Y_opt.copy()
        Y_opt.resize((max_iter, 1))
        T_eval = T_eval.copy()
        T_eval.resize((max_iter, 1))
        T_opt = T_opt.copy()
        T_opt.resize((max_iter, 1))
        failed_opt = failed_opt.copy()
        failed_opt.resize((max_iter, 1))
    # Don't load results - initialise arrays
    else:
        cur_iter = 0

        X_opt = np.zeros((max_iter, X_all.shape[1]))
        Y_opt = np.zeros((max_iter, 1))
        T_eval = np.zeros((max_iter, 1))
        T_opt = np.zeros((max_iter, 1))
        failed_opt = np.zeros((max_iter, 1))

    if random_state is not None:
        np.random.seed(random_state)

    for cur_iter in range(cur_iter, max_iter):
        t_0 = time()
        indices_notfailed = np.flatnonzero(1 - failed_opt[:cur_iter])
        X_potential = np.array(sorted(X_unexplored))

        # Select n_initial random solutions that don't fail
        if indices_notfailed.shape[0] < n_initial:
            decision_method = 'RAND'
            index = np.random.randint(X_potential.shape[0])
            params = X_potential[index, :]
        else: # Use GP + EI to select the next solution
            decision_method = 'GP+EI'
            # Select training data that didn't fail
            X_notfailed = X_opt[indices_notfailed, :]
            Y_notfailed = Y_opt[indices_notfailed]

            gp_target = GaussianProcess(X_notfailed, Y_notfailed, noiseless=True)
            gp_kern = GPy.kern.Bias(X_opt.shape[1]) + GPy.kern.Matern52(X_opt.shape[1], ARD=False)
            gp_notfail = GPy.models.GPClassification(X_opt[:cur_iter], (1-failed_opt[:cur_iter]), kernel=gp_kern.copy())
            gp_notfail.optimize_restarts(num_restarts=5, robust=True, verbose=False)

            ei = gp_target.expected_improvement(X_potential).ravel()
            prob_success = gp_notfail.predict(X_potential)[0].ravel()
            target = ei * prob_success

            if use_time:
                gp_time = GPy.models.GPRegression(X_opt, np.log(T_eval), kernel=gp_kern.copy())
                gp_time.optimize_restarts(num_restarts=5, robust=True, verbose=False)
                time_pred = np.exp(gp_time.predict(X_potential)[0]).ravel()
                if cur_iter >= 10: # Add the time taken by the optimiser itself.
                    time_pred += T_opt[cur_iter-10:cur_iter].mean()
                target /= time_pred

            target_index = np.argmax(target)
            params = X_potential[target_index]

        param_dict = dict(zip(parameter_ranges.keys(), params))
        print('Iteration {0}, {1}, params = {2}'.format(cur_iter + 1, decision_method, params), end='\t')

        t_1 = time()
        T_opt[cur_iter] = t_1 - t_0

        try:
            y_ = error_function(**param_dict)
        except tf.errors.ResourceExhaustedError as e:
        #except ValueError as e:
            t_e = time() - t_1
            print('FAILED!, time={0:.2f}s'.format(t_e))
            T_eval[cur_iter] = t_e
            failed_opt[cur_iter] = 1
        else:
            t_e = time() - t_1
            print('error = {0:.3f}, time = {1:.2f}s'.format(y_, t_e))
            Y_opt[cur_iter] = y_
            T_eval[cur_iter] = t_e
        X_opt[cur_iter, :] = params
        X_unexplored.remove(tuple(params))

    return X_opt, Y_opt, T_eval, T_opt, failed_opt
