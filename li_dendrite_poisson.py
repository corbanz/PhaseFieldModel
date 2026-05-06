"""
li_dendrite_poisson.py  (v2 — corrected sign convention)
=========================================================

Implements equation 15 of Yan et al. with a spatially varying η_a
from the Poisson solve (eq. 13 of Yan et al.).

Sign convention fix from v1
---------------------------
Equation 15 of Yan et al.:

    ∂ξ/∂t = −Mσ[W h'(ξ) − κ∇²ξ]
             − Mη · h'(ξ) · C · { exp[(1-α)nFη_a/RT]
                                 − exp[−α nF η_a/RT] }

The minus in front of Mη means: for the bracket to drive DEPOSITION
(∂ξ/∂t > 0 at interface), the bracket must be NEGATIVE.

The bracket is negative only when η_a < 0. That is the cathodic
convention: η_a < 0 drives Li⁺ + e⁻ → Li (deposition).

We define:
    η_a(x,y) = V_app − φ(x,y) − E°

Setting E° > V_app ensures η_a < 0 everywhere, driving deposition.
The spatial variation of φ from the Poisson solve then gives spatial
variation in η_a — the key addition over li_dendrite_pf.py.

Primary tip-enhancement mechanism
-----------------------------------
The Poisson solve makes φ slightly higher in the electrolyte near a
protruding tip (the high-conductivity electrode protrusion carries
V_app closer to the bulk). This makes η_a slightly LESS negative at
the tip (weaker electric driving force there).

However, the Li⁺ concentration C is HIGHER at the tip (it protrudes
into less-depleted electrolyte). In the BV bracket, the second term
C·exp[−αnFη_a/RT] is larger when C is larger, making the bracket
more negative — faster deposition at the tip. This concentration-
gradient mechanism (Mullins-Sekerka instability) is the primary
driver of dendritic branching in this model.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os

# -----------------------------------------------------------------------
# 1. Physical constants
# -----------------------------------------------------------------------
F_const = 96485.33212
Rgas    = 8.314
n_elec  = 1

# -----------------------------------------------------------------------
# 2. Grid
# -----------------------------------------------------------------------
Nx, Ny = 150, 150
Lx, Ly = 1.0, 1.0
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# -----------------------------------------------------------------------
# 3. Phase-field parameters  (Yan et al. eq. 15)
# -----------------------------------------------------------------------
kappa0  = 1.5e-4
W       = 1.0
M_sigma = 1.0
M_eta   = 0.8       # tuned so BV term is competitive with curvature

delta_aniso = 0.15  # anisotropy strength (Yan et al. Table 1)
mode        = 4     # 4-fold symmetry

# -----------------------------------------------------------------------
# 4. Electrochemical parameters
# -----------------------------------------------------------------------
alpha = 0.5
V_app = 0.10        # applied voltage (left BC of Poisson solve)
E_std = 0.18        # E° > V_app ensures η_a < 0 everywhere → deposition

sigma_e = 1.0e4     # electrode conductivity (high — Li metal)
sigma_s = 1.0       # electrolyte conductivity

# -----------------------------------------------------------------------
# 5. Temperature-dependent diffusion  (Yan et al. eq. 25)
# -----------------------------------------------------------------------
D0 = 5.0e3
Ea = 22.0e3         # J/mol

# -----------------------------------------------------------------------
# 6. Spatial operators
# -----------------------------------------------------------------------
def laplacian(u):
    up  = np.pad(u, ((1,1),(1,1)), mode="edge")
    uxx = (up[1:-1,2:] - 2*up[1:-1,1:-1] + up[1:-1,:-2]) / dx**2
    uyy = (up[2:,1:-1] - 2*up[1:-1,1:-1] + up[:-2,1:-1]) / dy**2
    return uxx + uyy

def gradients(u):
    up = np.pad(u, ((1,1),(1,1)), mode="edge")
    ux = (up[1:-1,2:] - up[1:-1,:-2]) / (2*dx)
    uy = (up[2:,1:-1] - up[:-2,1:-1]) / (2*dy)
    return ux, uy

# -----------------------------------------------------------------------
# 7. Anisotropic κ(θ)
# -----------------------------------------------------------------------
def aniso_kappa(xi):
    ux, uy = gradients(xi)
    theta  = np.arctan2(uy, ux)
    return kappa0 * (1.0 + delta_aniso * np.cos(mode * theta))**2

# -----------------------------------------------------------------------
# 8. Phase-field functions
# -----------------------------------------------------------------------
def g_prime(xi):
    """g'(ξ) = 2ξ(1-ξ)(1-2ξ)   double-well derivative"""
    return 2.0 * xi * (1.0 - xi) * (1.0 - 2.0 * xi)

def h_prime(xi):
    """h'(ξ) = 30ξ²(1-ξ)²   interface-localising interpolant
    Nonzero only at the electrode-electrolyte interface (0 < ξ < 1)."""
    return 30.0 * xi**2 * (1.0 - xi)**2

def h_interp(xi):
    """h(ξ) = ξ³(10 - 15ξ + 6ξ²)   conductivity interpolant"""
    return xi**3 * (10.0 - 15.0*xi + 6.0*xi**2)

# -----------------------------------------------------------------------
# 9. Poisson solver — Yan et al. eq. 13:  ∇·(σ(ξ)∇φ) = 0
# -----------------------------------------------------------------------
def solve_poisson(xi):
    """
    Solve for electrostatic potential φ given current phase field ξ.

    σ(ξ) = σ_s + (σ_e − σ_s)·h(ξ)
    BCs:  φ = V_app (left/electrode),  φ = 0 (right/bulk electrolyte)
          ∂φ/∂n = 0 (top and bottom — Neumann)
    """
    h     = h_interp(np.clip(xi, 0.0, 1.0))
    sigma = sigma_s + (sigma_e - sigma_s) * h

    # Face-averaged conductivities (finite-volume)
    sx = 0.5 * (sigma[:, :-1] + sigma[:, 1:])    # Ny × (Nx-1)
    sy = 0.5 * (sigma[:-1, :] + sigma[1:, :])    # (Ny-1) × Nx

    N   = Ny * Nx
    idc = lambda i, j: i * Nx + j
    rows, cols, vals = [], [], []
    b   = np.zeros(N)

    for i in range(Ny):
        for j in range(Nx):
            k = idc(i, j)
            if j == 0:                       # left  — Dirichlet
                rows.append(k); cols.append(k); vals.append(1.0)
                b[k] = V_app; continue
            if j == Nx - 1:                  # right — Dirichlet
                rows.append(k); cols.append(k); vals.append(1.0)
                b[k] = 0.0;    continue
            # interior + Neumann top/bottom
            d = 0.0
            for (si, sj, sf, h2) in [
                    ( 0,+1, sx[i,j],   dx**2),   # right face
                    ( 0,-1, sx[i,j-1], dx**2),   # left  face
                    (+1, 0, sy[i,j]   if i < Ny-1 else 0, dy**2),  # up
                    (-1, 0, sy[i-1,j] if i > 0    else 0, dy**2),  # down
            ]:
                if sf == 0: continue
                c_ = sf / h2
                rows.append(k); cols.append(idc(i+si, j+sj)); vals.append(c_)
                d -= c_
            rows.append(k); cols.append(k); vals.append(d)

    A   = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    phi = spla.spsolve(A, b).reshape(Ny, Nx)
    return phi

def compute_eta_a(phi):
    """
    η_a(x,y) = V_app − φ(x,y) − E°

    With E° > V_app:  η_a < 0 everywhere in the electrolyte.
    Negative η_a drives the cathodic (deposition) reaction in eq. 15.
    Spatial variation of φ → spatial variation of η_a.
    """
    return V_app - phi - E_std

# -----------------------------------------------------------------------
# 10. Butler-Volmer rate  (the {} bracket in Yan et al. eq. 15)
# -----------------------------------------------------------------------
def butler_volmer(C, eta_a, T):
    """
    R_BV = exp[(1-α)nFη_a/RT]  −  C · exp[−αnFη_a/RT]

    With η_a < 0:
      first term  < 1  (anodic direction suppressed)
      second term > C  (cathodic direction amplified)
      R_BV < 0

    Then in eq. 15: −Mη · h'(ξ) · C · R_BV > 0  →  ξ increases
    i.e. deposition occurs. ✓

    Larger |C| at the tip → R_BV more negative → faster deposition.
    That is the concentration-driven (Mullins-Sekerka) instability.
    """
    fac   = n_elec * F_const / (Rgas * T)
    exp_a = np.exp(np.clip( (1-alpha)*fac*eta_a, -50, 50))
    exp_b = np.exp(np.clip(-alpha    *fac*eta_a, -50, 50))
    return exp_a - C * exp_b

# -----------------------------------------------------------------------
# 11. Initial conditions & BCs
# -----------------------------------------------------------------------
def initial_conditions():
    xi = np.zeros((Ny, Nx))
    c  = np.ones( (Ny, Nx))
    xi[:, :4] = 1.0;  c[:, :4] = 0.0     # flat electrode
    yy, xx = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing="ij")
    seed = np.exp(-((xx - 12)**2 / 25.0 + (yy - Ny//2)**2 / 9.0))
    xi   = np.maximum(xi, 0.98*seed)
    c    = np.minimum(c,  1.0-0.98*seed)
    return xi, c

def apply_bc(xi, c):
    xi[:,0]=1.0;  c[:,0]=0.0
    xi[:,-1]=0.0; c[:,-1]=1.0
    return xi, c

# -----------------------------------------------------------------------
# 12. Time-stepping
# -----------------------------------------------------------------------
def run_simulation(T_kelvin=298.0, nsteps=25000,
                   save_every=2500, poisson_every=25,
                   verbose=True):
    xi, c = initial_conditions()
    D_T   = D0 * np.exp(-Ea / (Rgas * T_kelvin))
    dt    = min(0.2*min(dx,dy)**2/(D_T+kappa0*(1+delta_aniso)**2+1e-30),
                5e-5)

    phi   = solve_poisson(xi)
    eta_a = compute_eta_a(phi)
    snaps = [(0, xi.copy(), c.copy(), phi.copy(), eta_a.copy())]

    for step in range(1, nsteps+1):
        if step % poisson_every == 0:
            phi   = solve_poisson(xi)
            eta_a = compute_eta_a(phi)

        R_bv = butler_volmer(c, eta_a, T_kelvin)
        kap  = aniso_kappa(xi)

        # Equation 15 of Yan et al.
        dxi = (-M_sigma*(W*g_prime(xi) - kap*laplacian(xi))
               - M_eta*h_prime(xi)*R_bv)
        dc  =  D_T*laplacian(c) - h_prime(xi)*R_bv

        xi += dt*dxi;  c += dt*dc
        xi  = np.clip(xi,0,1);  c = np.clip(c,0,1)
        xi, c = apply_bc(xi, c)

        if step % save_every == 0:
            phi_s = solve_poisson(xi)
            eta_s = compute_eta_a(phi_s)
            snaps.append((step, xi.copy(), c.copy(), phi_s, eta_s))
            if verbose:
                iface = (xi>0.05)&(xi<0.95)
                tip   = int((xi>0.5).any(axis=0).nonzero()[0].max()) \
                        if (xi>0.5).any() else 0
                print(f"  T={T_kelvin-273.15:.0f}°C  step {step:5d}  "
                      f"tip={tip}/{Nx-1}  "
                      f"η_a∈[{eta_s[iface].min():.3f},"
                      f"{eta_s[iface].max():.3f}]  "
                      f"C_tip={c[Ny//2,tip]:.3f}")
    return snaps, dt, D_T

# -----------------------------------------------------------------------
# 13. Plotting helper
# -----------------------------------------------------------------------
def plot_all_fields(snaps, title, outpath):
    n = len(snaps)
    fig, axes = plt.subplots(4, n, figsize=(3.5*n, 13))
    labels = [r'Phase field $\xi$',
              r'Li$^+$ concentration $C$',
              r'Potential $\phi$',
              r'Overpotential $\eta_a$']
    cmaps  = ['viridis', 'magma', 'plasma', 'RdBu_r']
    for col, (step, xi, c, phi, eta_a) in enumerate(snaps):
        for row, (fld, lbl, cm) in enumerate(
                zip([xi,c,phi,eta_a], labels, cmaps)):
            ax = axes[row,col]
            im = ax.imshow(fld, origin='lower', cmap=cm, aspect='equal')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            if row==0: ax.set_title(f'step {step}', fontsize=10)
            if col==0: ax.set_ylabel(lbl, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()

# -----------------------------------------------------------------------
# 14. Driver
# -----------------------------------------------------------------------
if __name__ == "__main__":

    # Single temperature — four fields over time
    T_run, nsteps = 298.15, 10000
    print(f"=== Single run  T={T_run-273.15:.0f}°C  ({nsteps} steps) ===")
    snaps, dt, D_T = run_simulation(T_kelvin=T_run, nsteps=nsteps,
                                    save_every=2500, poisson_every=25)
    plot_all_fields(
        snaps,
        f'Li dendrite — spatially varying $\\eta_a$   '
        f'T={T_run-273.15:.0f}°C   D(T)={D_T:.2e}\n'
        f'$\\eta_a = V_{{app}} - \\phi - E^\\circ$  '
        f'(negative = cathodic = deposition)',
        'poisson_fields.png')

    # Temperature sweep
    print("\n=== Temperature sweep ===")
    fig2, axes2 = plt.subplots(2, 3, figsize=(13, 8))
    for col, T in enumerate([273.15, 298.15, 323.15]):
        print(f"  T={T-273.15:.0f}°C ...")
        s, _, D_T = run_simulation(T_kelvin=T, nsteps=nsteps,
                                   save_every=nsteps, poisson_every=25,
                                   verbose=False)
        _, xi, c, phi, eta_a = s[-1]
        ax = axes2[0,col]
        ax.imshow(xi, origin='lower', vmin=0, vmax=1, cmap='viridis')
        ax.set_title(f'T={T-273.15:.0f}°C   D={D_T:.2e}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax = axes2[1,col]
        im = ax.imshow(eta_a, origin='lower', cmap='RdBu_r')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(r'$\eta_a$ — blue=more cathodic', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes2[0,0].set_ylabel(r'Phase field $\xi$', fontsize=10)
    axes2[1,0].set_ylabel(r'Overpotential $\eta_a$', fontsize=10)
    fig2.suptitle(r'Temperature sweep — corrected cathodic $\eta_a$',
                  fontsize=12)
    plt.tight_layout()
    plt.savefig('poisson_fields.png',
                dpi=130, bbox_inches='tight')
    print("Saved: poisson_sweep.png")

    if not os.environ.get('MPLBACKEND','').lower()=='agg':
        plt.show()
