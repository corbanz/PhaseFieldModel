import numpy as np
import matplotlib.pyplot as plt

# Grid
Nx, Ny = 256, 256
Lx, Ly = 1.0, 1.0
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# Parameters
M = 1.0
kappa = 1.0e-4

# Time stepping
dt = 0.20 * min(dx, dy)**2 / (M * kappa + 1e-30) 
nsteps = 4000
plot_every = 100

# Initial condition
rng = np.random.default_rng(0)
xi = 0.5 + 0.02 * rng.standard_normal((Ny, Nx))
xi = np.clip(xi, 0.0, 1.0)

def laplacian_neumann(u):

    up = np.pad(u, ((1, 1), (1, 1)), mode="edge")
    uxx = (up[1:-1, 2:] - 2*up[1:-1, 1:-1] + up[1:-1, :-2]) / dx**2
    uyy = (up[2:, 1:-1] - 2*up[1:-1, 1:-1] + up[:-2, 1:-1]) / dy**2
    return uxx + uyy

def hprime(x):
    return 2*x*(1-x)*(1-2*x)

# Plot setup
plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(xi, origin="lower", vmin=0, vmax=1)
ax.set_title("Allen–Cahn phase field ξ")
plt.colorbar(im, ax=ax)

for step in range(1, nsteps + 1):
    lap = laplacian_neumann(xi)
    xi_t = M * (kappa * lap - hprime(xi))
    xi = xi + dt * xi_t

    # Keep bounded
    xi = np.clip(xi, 0.0, 1.0)

    if step % plot_every == 0:
        im.set_data(xi)
        ax.set_xlabel(f"step={step}, dt={dt:.2e}")
        plt.pause(0.001)

plt.ioff()
plt.show()