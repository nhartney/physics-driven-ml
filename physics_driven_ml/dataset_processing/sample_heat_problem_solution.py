"""
Solve the time-dependent heat problem with forcing to experiment with initial and
boundary condition choices.
"""
from firedrake import (Constant, Function, TestFunction, FunctionSpace,
                       SpatialCoordinate, RectangleMesh, sin, pi, exp,
                       inner, grad, dx, VTKFile, DirichletBC, solve, project)
from numpy import random

# Test parameters
k = Constant(1.)
dt = Constant(0.001)
ntimesteps = 10

# Domain
Lx = Ly = 1
nx = ny = 5
mesh = RectangleMesh(nx, ny, Lx, Ly, name="mesh")
x,y = SpatialCoordinate(mesh)

# Output
outfile = VTKFile("sample_heat_solution.pvd")

# Set up problem
V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
u_ = Function(V)
v = TestFunction(V)
x,y = SpatialCoordinate(mesh)

# Initial conditions and boundray conditions
# a controls the amplitude of the Gaussian (between 1 and 2)
a =  1 + random.rand()
# x_pos and y_pos control the position of the initial Gaussian (between 0 and 1)
x_pos = random.rand()
y_pos = random.rand()
IC = a*exp(-((x-x_pos)**2)/0.01-((y-y_pos)**2)/0.01)
u_.interpolate(IC)
bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]

# Timestep
for n in range(ntimesteps):
    time = n*dt
    f = Function(V).interpolate(u*time*sin(pi*x)*sin(pi*y))
    F = (inner((u - u_)/dt, v) + inner(k * grad(u), grad(v)) - inner(f, v)) * dx
    # Solve PDE (using LU factorisation)
    solve(F == 0, u, bcs=bcs)    
    u_.assign(u)
    outfile.write(project(u, V, name="u_solution"), project(f, V, name="f"))