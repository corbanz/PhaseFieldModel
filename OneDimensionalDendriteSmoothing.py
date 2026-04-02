import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# 1D Phase-Field Model — Electrode/Electrolyte Interface
# PURPOSE: Reproduce the red curve (xi) in Fig. 4 of the thermal dendrite paper.
# xi = 1 on the left (solid Li electrode), a smooth S-shaped transition in the middle (the interface), and xi = 0 on the right (liquid electrolyte).
# EQUATION (first term of Eq. 15 from the paper):d(xi)/dt = M_xi * (kappa * d²xi/dx² - h'(xi))
# kappa * d²xi/dx² and h'(xi) compete to produce a stable diffuse interface
# normalized (dimensionless) units like the paper
# The domain represents ~800 micrometers (8 * l0, l0 = 100 um)
# =============================================================================
N=400 # number of grid points along x
L=8.0 # domain length in normalized units
dx=L/(N-1) # spacing between grid points
x=np.linspace(0, L, N) # array of x positions using np.linspace(start, stop, num)
# =============================================================================
# Parameters:
# Table 1 in Chen et al.
# M_xi = interfacial mobility (La in Chen, Ma in Yan) = 2000 normalized in papers. Controls how fast the interface moves and relaxes
# kappa = gradient energy coefficient (κ0 in Yan) = .01 normalized. Interface width.
# W = Barrier height (W in Yan) = .25 normalized
M_xi=1.0 #this is just speed so it doesn't need to be fast, so 2000 -> 1
kappa=0.01 
W=0.25
# =============================================================================
#Stepping
#Use explicit euler time stepping "xi_new = xi_old + dt * (rate of change)"
# stable time step for this equation scales with dx^2 and use a safety factor of .4 to keep stable, .5 is the maximum possible use
# M_xi * kappa is the diffusion coefficient and how fast something spreads out 
# The grid cannot spread faster than one grid spacing (dx) in one time step (dt) -  dt ≤ dx² / (2 * M_xi * kappa) - Von Neumann stability condition
dt=0.4 * dx**2 / (M_xi * kappa) # dt ≤ dx² / (2 * M_xi * kappa), .4<.5
nsteps=8000 #some large number of steps just to reach equilibrium
plot_every=500 #how often the plot is updated
# =============================================================================
# Initial Condition
# x_i = 1 to represent the electrode (solid) and 0 to represent the electrolyte
# The simulation smooths it over time
interface_pos = L/2 # interface is at x=4 (L/2) which is the midpt of the domain
xi = np.where(x < interface_pos, 1.0, 0.0) #checks where x is less than the interface position (which means less that L/2), if yes: =1, if no: =0
# =============================================================================
# Helper Functions
def laplacian_1d(u): #d²ξ/dx²
    u_pad = np.pad(u, (1, 1), mode='edge')
    return (u_pad[2:] - 2*u_pad[1:-1] + u_pad[:-2]) / dx**2 #some fancy way to just calculate the second derivative, not totally sure how
""" Takes a ξ point and determines if its positive, negatice, or zero. If positive, ξ is lower than nearby grid points (valley)
and that point is pushed upwards. If negative, ξ is higher than nearby points (peak) and is pushed down. If zero, nothing.
 https://en.wikipedia.org/wiki/Laplace_operator """
def hprime(xi): 
    return 2 * W * xi * (1 - xi) * (1 - 2*xi) 
"""Derivative of the double well potential g(ξ) from Chen [26]
g(ξ) = W * ξ² * (1-ξ)²
g'(ξ) = 2W * ξ * (1-ξ) * (1-2ξ)
There is minima as ξ=0 and ξ=1 which are stable and the hill at ξ=.5 which is unstable
There is no slope at the endpoints. There is no slope at ξ=.5 as it is a local maxima, so any change left or right will slope it towards
either 0 or 1
Pos-slope between .5 and 1, negative between .5 and 0
"""
# =============================================================================
# Plot setup
plt.ion() #interactive mode for matplotlib, live updating
fig, ax = plt.subplots(figsize=(9, 5)) #figure window creation
line, = ax.plot(x, xi, 'r-', linewidth=2, label='ξ (phase field)')
ax.axvline(interface_pos, color='gray', linestyle='--', alpha=0.4,
           label='Initial interface position')
ax.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
ax.axhline(0.0, color='gray', linestyle=':', alpha=0.3)
 
ax.set_xlim(0, L)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel('Position (normalized, 1 unit = 100 μm)')
ax.set_ylabel('Phase field ξ')
ax.set_title('1D Phase Field — Electrode | Interface | Electrolyte')
ax.legend(loc='upper right')
 
# Label the regions
ax.text(1.0, 0.5, 'ELECTRODE\n(ξ = 1)\nSolid Li',
        ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
ax.text(6.5, 0.5, 'ELECTROLYTE\n(ξ = 0)\nLiquid',
        ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
 
# -----------------------------------------------------------------------------
# TIME INTEGRATION LOOP
# Each step: compute the Laplacian and double-well derivative, then
# update xi using forward Euler: xi_new = xi + dt * (dxi/dt)
# -----------------------------------------------------------------------------
for step in range(1, nsteps + 1):
 
    lap  = laplacian_1d(xi)               # curvature term
    dxdt = M_xi * (kappa * lap - hprime(xi))  # full Allen-Cahn RHS
    xi   = xi + dt * dxdt                 # forward Euler update
    xi   = np.clip(xi, 0.0, 1.0)         # keep xi physically bounded
 
    if step % plot_every == 0:
        line.set_ydata(xi)
        ax.set_title(f'1D Phase Field  |  step = {step}  |  dt = {dt:.4f}')
        plt.pause(0.01)
 
plt.ioff()
 
# -----------------------------------------------------------------------------
# FINAL PLOT — equilibrium profile
# This S-shaped curve going from 1 (left) to 0 (right) is what the red
# curve in Fig. 4 of the paper represents.
# -----------------------------------------------------------------------------
ax.set_title('1D Phase Field — Equilibrium Interface Profile (Final)')
line.set_label('ξ — equilibrium profile')
ax.legend()
plt.tight_layout()
plt.show()
 
print(f"\nSimulation complete.")
print(f"Grid points : {N}")
print(f"Domain size : {L} (normalized units)")
print(f"dx          : {dx:.4f}")
print(f"dt          : {dt:.6f}")
print(f"Total steps : {nsteps}")
print(f"\nThe interface has settled into a smooth diffuse profile.")
print(f"xi ranges from {xi.min():.4f} (electrolyte) to {xi.max():.4f} (electrode)")
print(f"\nNext step: add Li+ concentration diffusion (C+) on top of this profile.")
 