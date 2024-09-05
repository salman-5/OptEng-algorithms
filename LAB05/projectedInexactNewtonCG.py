# Optimization for Engineers - Dr.Johannes Hild
# projected inexact Newton descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dH = projectedHessApprox(f, P, x, d) from projectedHessApprox.py
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

import numpy as np
import projectedBacktrackingSearch as PB
import projectedHessApprox as PHA

def matrnr():
    matrnr = 23391770
    return matrnr


def projectedInexactNewtonCG(f, P, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start projectedInexactNewtonCG...')

    countIter = 0
    xp= P.project(x0)
    etak = min(0.5, np.sqrt(np.linalg.norm(xp - P.project(xp - f.gradient(xp)))) * np.linalg.norm(xp - P.project(xp - f.gradient(xp))))
    while np.linalg.norm(xp - P.project(xp - f.gradient(xp))) > eps:
        xj=xp.copy()
        rj = f.gradient(xp)
        dj = -rj.copy()        
        loop_break=0
        while np.linalg.norm(rj) > etak:
            dA = PHA.projectedHessApprox(f, P, xp, dj)
            rhoj = dj.T @ dA
            if rhoj <= eps * np.linalg.norm(dj)**2:
                break
            
            tj = (np.linalg.norm(rj)**2) / rhoj
            xj= xj + tj*dj
            r_old = rj.copy()
            rj= r_old + tj*dA
            beta = (np.linalg.norm(rj)**2) / (np.linalg.norm(r_old)**2)
            dj= -rj+ beta * dj
            loop_break+=1
        if loop_break==0:
            dk=-f.gradient(xp)
        else:
            dk = xj - xp

        tk = PB.projectedBacktrackingSearch(f, P, xp, dk)
        xp = P.project(xp + tk*dk)
        etak = min(0.5, np.sqrt(np.linalg.norm(xp - P.project(xp - f.gradient(xp)))) * np.linalg.norm(xp - P.project(xp - f.gradient(xp))))
        countIter += 1

    if verbose:
        gradx = f.gradient(xp)
        stationarity = np.linalg.norm(xp - P.project(xp - gradx))
        print('projectedInexactNewtonCG terminated after ', countIter, ' steps with stationarity =', np.linalg.norm(stationarity))

    return xp

