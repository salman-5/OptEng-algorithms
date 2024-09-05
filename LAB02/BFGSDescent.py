# Optimization for Engineers - Dr.Johannes Hild
# global BFGS descent

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + t_k * d_k
# d_k is the BFGS direction. If a descent direction check fails, d_k is set to steepest descent and the inverse BFGS matrix is reset.
# t_k results from Wolfe-Powell

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# t = WolfePowellSearch(f, x, d) from WolfePowellSearch.py

# Test cases:
# myObjective = noHessianObjective()
# x0 = np.array([[-0.01], [0.01]])
# xmin = BFGSDescent(myObjective, x0, 1.0e-6, 1)
# should return
# xmin close to [[0.26],[-0.21]] with the inverse BFGS matrix being close to [[0.0078, 0.0005], [0.0005, 0.0080]]


import numpy as np
import WolfePowellSearch as WP


def matrnr():
    matrnr = 23391770
    return matrnr


def BFGSDescent(f, x0: np.array, eps=1.0e-3, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start BFGSDescent...')

    countIter = 0
    x = x0
    n = x0.shape[0]
    E = np.eye(n)
    B = E
    def inverse_BK(B,delta_xk,delta_gk):
        rk=delta_xk-B@delta_gk
        B=B+(rk@delta_xk.T+delta_xk@rk.T)/(delta_gk.T@delta_xk)-((rk.T@delta_gk)/(delta_gk.T @ delta_xk)**2)*(delta_xk@delta_xk.T)
        return B
    while(np.linalg.norm(f.gradient(x))>eps):
        dk=-B@f.gradient(x)
        descent = f.gradient(x).T @ dk

        if(descent>=0):
            dk=-f.gradient(x)
            B=np.eye(n)
        tk=WP.WolfePowellSearch(f,x,dk)
        delta_gk=f.gradient(x+tk*dk)-f.gradient(x)
        delta_xk=tk*dk
        x=x+tk*dk
        if((delta_gk.T @delta_xk)<=0):
            B=np.eye(n)
        else:
            B=inverse_BK(B,delta_xk,delta_gk)
        countIter=countIter+1

    if verbose:
        gradx = f.gradient(x)
        print('BFGSDescent terminated after ', countIter, ' steps with norm of gradient =', np.linalg.norm(gradx), 'and the inverse BFGS matrix is')
        print(B)

    return x
