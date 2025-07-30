from firedrake import (FunctionSpace, Constant, Function, TestFunction,
                       DirichletBC, inner, grad, dx, solve)


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

    def timestep(self, ntimesteps, u_ic):
        self.u_.assign(u_ic)
        for n in range(ntimesteps):
            # Solve PDE (using LU factorisation)
            solve(self.residual == 0, self.u, bcs=self.bcs)
            self.u_.assign(self.u)
        return self.u
    
