
import ast
from io import StringIO
import numpy as np
from parse import Encoder
from pysmt.smtlib.parser import get_formula as stream_to_pysmt
from pysmt.shortcuts import And, Exp, Not, LE, LT, Ite, Minus
from pysmt.shortcuts import Plus, Pow, REAL, Real, Symbol, Times
from pywmi import Domain, Density
from scipy.stats import norm
from z3 import Solver


def z3_to_pysmt(f):
    solver = Solver()
    solver.add(f)
    z3str = solver.to_smt2()
    return stream_to_pysmt(StringIO(z3str))


def weight_to_pysmt(vdist, epsilon=1e-3):

    factors = []
    bounds = []
    for var, dist in vdist.items():
        smtvar = Symbol(str(var), REAL)
        if dist[0] == 'G':
            mean, variance = dist[1], dist[2]
            dist = Times(Real(float(1/np.sqrt(variance*2*np.pi))),
                         Exp(Times(Real(float(-1/(2*variance))),
                                   Pow(Minus(smtvar, Real(float(mean))),
                                       Real(2)))))
            # bounds on Normal RVs are computed independently
            b = norm.ppf(epsilon/2, loc=mean, scale=np.sqrt(variance))
            lb = float(b)
            ub = float(2*mean - b)

        elif dist[0] == 'S':
            steps = []
            xmin = np.infty
            xmax = -np.infty
            for x1, x2, p in dist[1]:
                cond = And(LE(Real(x1), smtvar), LT(smtvar, Real(x2)))
                steps.append(Ite(cond, Real(p), Real(0)))
                xmin = np.min([xmin, x1])
                xmax = np.max([xmax, x2])

            dist = Plus(*steps)
            lb = float(xmin)
            ub = float(xmax)

        else:
            raise NotImplementedError(f"Unsupported dist: {dist[0]}")

        factors.append(dist)
        bound = And(LE(Real(lb), smtvar),
                    LE(smtvar, Real(ub)))
        bounds.append(bound)

    return And(*bounds), Times(*factors)
            
            
    

    


if __name__ == '__main__':

    from sys import argv

    if len(argv) != 2:
        print("USAGE: python3 to_pywmi PATH")
        exit(1)

    path = argv[1]
    with open(path, 'r') as f:
        node = ast.parse(f.read())

    p = Encoder()
    p.visit(node)
    
    program = And(z3_to_pysmt(p.model),
                  z3_to_pysmt(p.program))

    bounds, weight = weight_to_pysmt(p.vdist)
    support = And(program, bounds)

    minority = z3_to_pysmt(p.sensitiveAttribute)
    hired = z3_to_pysmt(p.fairnessTarget)

    queries = [minority, And(minority, hired), And(Not(minority), hired)]

    bvars = []
    cvars = []
    cbounds = []
    
    domain = Domain.make(bvars, cvars, cbounds)
    density = Density(domain, support, weight, queries)

    density_path = path + '_density.json'
    density.to_file(density_path)


