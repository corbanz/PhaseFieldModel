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
# 