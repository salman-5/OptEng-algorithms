# Optimization for Engineers - Dr.Johannes Hild
# Least squares model objective

# Purpose: Provides .residual() and .jacobian() of the least squares mapping p -> 0.5*sum_k (model(xData_k,p)-fData_k)**2

# Input Definition:
# model: objective class with methods .objective() and .gradient() for data evaluation
# and .setParameters() and .parameterGradient()
# p: column vector in R**m (parameter space)
# xData: matrix in R**nxN (measure points). xData[:,k].reshape((n,1)) returns the k-th measure point as column vector.
# fData: row vector in R**1xN (measure results). fData[:,k] returns the k-th measure result as a scalar.

# Output Definition:
# residual(): column vector in R**N, the k-th entry is model(xData_k,p)-fData_k
# jacobian(): matrix in R**Nxm, the [k,j]-th entry returns: partial derivative with respect to p_j of (model(xData_k,p)-fData_k)

# Required files:
# <none>

# Test cases:

import numpy as np
from simpleValleyObjective  import simpleValleyObjective

def matrnr():
    matrnr = 23391770
    return matrnr


class leastSquaresModel:

    def __init__(self, model, xData: np.array, fData: np.array):
        self.model = model
        self.xData = xData
        self.fData = fData
        self.N = fData.shape[1]
        self.n = xData.shape[0]

    def residual(self, p: np.array):
        self.model.setParameters(p)
        myResidual = np.zeros((self.N, 1))
        model_values = np.apply_along_axis(
            lambda x: self.model.objective(x.reshape(-1, 1)), 0, self.xData)
        model_values = model_values -self.fData
        myResidual= model_values.reshape(self.N,1)
        return myResidual

    def jacobian(self, p: np.array):
        self.model.setParameters(p)
        myJacobian = np.zeros((self.N, p.shape[0]))
        jacobian_columns = np.apply_along_axis(
            lambda x: self.model.parameterGradient(x.reshape(-1, 1)).reshape(1,-1), 0, self.xData)
        myJacobian = jacobian_columns.T.squeeze()
        return myJacobian

