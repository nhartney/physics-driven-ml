from firedrake import (FunctionSpace, Constant, Function, TestFunction,
                       DirichletBC, inner, grad, dx, solve, SpatialCoordinate,
                       sin, interpolate, pi, assemble)
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

    def __init__(self, mesh, dt, create_training_data=False, dirname=None, chkptfreq=None):
        domain = Domain(mesh, dt, "CG", 1)
        V = FunctionSpace(mesh, "CG", 1)
        self.V = V
        domain.spaces.add_space("CG", V)
        if create_training_data:
            checkpoint=True
            multichkpt=True,
            diagnostic_fields=[Gradient("q")]
        else:
            checkpoint = False
            multichkpt=False
            diagnostic_fields=None
        output = OutputParameters(dirname=dirname,
                                  dump_vtus=True,
                                  dump_nc=False,
                                  dump_diagnostics=False,
                                  checkpoint=checkpoint,
                                  chkptfreq=chkptfreq,
                                  multichkpt=multichkpt)
        io = IO(domain, output, diagnostic_fields=diagnostic_fields)
        params = DiffusionParameters(domain.mesh, kappa=1)
        self.eqn = DiffusionEquation(domain, V, "q", params)

        if create_training_data:
            # if we're creating the training data we need to add the forcing
            source = Function(V)
            self.q = Function(V)
            # current time is a constant (needs to be on a space for adjoint?)
            R = FunctionSpace(mesh, "R", 0)
            self.t = Function(R)
            x, y = SpatialCoordinate(mesh)
            self.source_interpolate = interpolate(self.q*sin(self.t + dt)*sin(pi*x)*sin(pi*y), source)
            label=PhysicsLabel("forcing_term")
            self.eqn.residual -= source_label(label(subject(prognostic(self.eqn.test * source * dx,
                                                                       "q"), self.q),
                                               self.evaluate))

        scheme = BackwardEuler(domain)
        diffusion_methods = [CGDiffusion(self.eqn, "q", params)]
        self.stepper = Timestepper(self.eqn, scheme, io,
                                   spatial_methods=diffusion_methods)

    def evaluate(self, x_in, dt):
        self.q.assign(self.stepper.fields("q"))
        self.t.assign(self.eqn.domain.t)
        assemble(self.source_interpolate)

    def advance(self, u_out, u_in, ndt):
        u0 = self.stepper.fields("q")
        u0.assign(u_in)
        tmax = float(self.stepper.dt * ndt)
        self.stepper.run(0, tmax)
        u_out.assign(self.stepper.fields("q"))
