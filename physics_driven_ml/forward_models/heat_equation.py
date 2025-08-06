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

    def advance(self, u_out, u_in, ndt):
        self.u_.assign(u_in)
        for n in range(ndt):
            # Solve PDE (using LU factorisation)
            solve(self.residual == 0, self.u, bcs=self.bcs)
            self.u_.assign(self.u)
        u_out.assign(self.u_)


class GustoHeatEquationModel(object):

    def __init__(self, mesh, dt, create_training_data=False):
        domain = Domain(mesh, dt, "CG", 1)
        V = FunctionSpace(mesh, "CG", 1)
        self.V = V
        domain.spaces.add_space("CG", V)
        output = OutputParameters(dirname="gusto_heat_equation",
                                  dump_vtus=False,
                                  dump_nc=False,
                                  dump_diagnostics=False)
        io = IO(domain, output)
        params = DiffusionParameters(domain.mesh, kappa=1)
        eqn = DiffusionEquation(domain, V, "f", params)

        if create_training_data:
            # if we're creating the training data we need to add the forcing
            f = Function(V)
            eqn.residual -= physics_label(prognostic(eqn.test * f * dx, "f"))

        scheme = BackwardEuler(domain)
        diffusion_methods = [CGDiffusion(eqn, "f", params)]
        self.stepper = Timestepper(eqn, scheme, io,
                                   spatial_methods=diffusion_methods)

    def advance(self, u_out, u_in, ndt):

        u0 = self.stepper.fields("f")
        u0.assign(u_in)
        tmax = float(self.stepper.dt * ndt)
        self.stepper.run(0, tmax)
        u_out.assign(self.stepper.fields("f"))
