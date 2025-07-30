from firedrake import (FunctionSpace, Constant, Function, TestFunction,
                       DirichletBC, inner, grad, dx, solve)
from gusto import *


class HeatEquation(object):

    def __init__(self, mesh, dt):
        self.mesh = mesh
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.dt = Constant(dt)
        self.setup_residual()

    def setup_residual(self):
        k = Constant(1)
        self.u = Function(self.V)
        self.u_ = Function(self.V)
        v = TestFunction(self.V)
        self.bcs = [DirichletBC(self.V, Constant(0.0), "on_boundary")]
        self.residual = ((inner((self.u - self.u_)/self.dt, v)
                          + inner(k * grad(self.u), grad(v))) * dx
                         )

    def advance(self, ntimesteps, u_ic):
        self.u_.assign(u_ic)
        for n in range(ntimesteps):
            # Solve PDE (using LU factorisation)
            solve(self.residual == 0, self.u, bcs=self.bcs)
            self.u_.assign(self.u)
        return self.u


class GustoHeatEquationModel(object):

    def __init__(self, mesh, dt):
        domain = Domain(mesh, dt, "CG", 1)
        V = FunctionSpace(mesh, "CG", 1)
        self.V = V
        domain.spaces.add_space("CG", V)
        output = OutputParameters(dirname="gusto_heat_equation")
        io = IO(domain, output)
        params = DiffusionParameters(domain.mesh, kappa=1)
        eqn = DiffusionEquation(domain, V, "f", params)
        diffusion_scheme = BackwardEuler(domain)
        diffusion_methods = [CGDiffusion(eqn, "f", params)]
        self.stepper = Timestepper(eqn, diffusion_scheme, io,
                                   spatial_methods=diffusion_methods)

    def advance(self, ntimesteps, u_ic):
        u0 = self.stepper.fields("f")
        u0.assign(u_ic)
        tmax = float(self.stepper.dt * ntimesteps)
        self.stepper.run(0, tmax)

        return u0
