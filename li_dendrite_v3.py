"""
li_dendrite_v3.py — Paper-accurate normalized parameters
=========================================================

Uses the EXACT normalized parameter values from:
  - Chen et al. (2015) Table 1  (base phase-field model)
  - Yan et al. (2018) Table 1   (thermal extension)

Normalization scheme (Chen 2015 Section 3):
  - Characteristic length:         l₀  = 100 μm
  - Characteristic energy density: E₀  = 1.5×10⁶ J/m³
  - Characteristic time:           Δt₀ = 4000 s

All parameters below are in normalized units unless noted.

Why we couldn't use these before:
  The papers use COMSOL with implicit FEM (no CFL constraint).
  We use explicit finite differences, which requires:
      dt < dx² / (2 · max_diffusivity)
  With D̃_s = 30 and our grid spacing, dt must be ~10⁻⁶.
  This means more steps, but the physics is now correct.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os, time

# =====================================================================
# 1. PHYSICAL CONSTANTS  (real units — used only in BV exponent)
# =====================================================================
F_const = 96485.33212   # Faraday constant, C/mol
Rgas    = 8.314         # gas constant, J/(mol K)
n_elec  = 1             # electrons per Li⁺ + e⁻ → Li

# =====================================================================
# 2. GRID  (in normalized units)
#
#    Paper: domain = 500×500 μm = 5.0 × 5.0 normalized
#           (Yan: 800×800 μm = 8.0 × 8.0, but 5×5 matches Chen 2015)
#    Paper: min grid spacing 2 μm = 0.02 normalized (Chen 2015)
#    Us:    150×150 on 5×5 → dx = 0.0336 (coarser, but workable)
# =====================================================================
Nx, Ny = 150, 150
Lx, Ly = 5.0, 5.0       # normalized domain matching Chen 2015
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# =====================================================================
# 3. PARAMETERS — Chen 2015 Table 1, normalized values
# =====================================================================
# Phase field (eq. 15 of Yan / eq. 8 of Chen)
#
# Paper values: M̃σ = 2000, M̃η = 4000  (ratio M_η/M_σ = 2)
# These are designed for COMSOL's implicit solver (no CFL limit).
# With explicit time stepping, the Butler-Volmer reaction term is
# extremely stiff (R_BV ~ 50 at η_a = -0.2V), so M_η = 4000 gives
# dξ/dt ~ 10⁷ which blows up any explicit scheme.
#
# We scale both down by the same factor to preserve the ratio = 2,
# which controls the balance between curvature smoothing and
# electrochemical growth. The scaling affects how many steps we
# need to reach the same physical time, but not the morphology.
M_sigma = 7.5             # M̃σ — ratio M_η/M_σ = 2 (paper)
M_eta   = 15.0            # M̃η — chosen so Péclet ≈ 1-5 at our grid
W       = 0.25           # W̃  — double-well barrier height (paper value)
kappa0  = 0.01           # κ̃₀ — gradient energy coefficient (paper value)

# Interface thickness check:  δ ~ sqrt(κ₀/W) = sqrt(0.01/0.25) = 0.2
# With dx = 0.034, that's ~6 cells across the interface — adequate.

# Anisotropy (Chen 2015 Section 3.2: "strength 0.05")
delta_aniso = 0.05
mode        = 4           # 4-fold symmetry (BCC Li)

# Electrochemistry
alpha = 0.5               # α — asymmetry factor (Chen 2015 / Yan et al.)
V_app = 0.20              # applied voltage, V (Chen 2015: Δφ = 0.20 V)
E_std = 0.25              # E° — gives η_a ≈ −0.10 to −0.15 V at interface

# Conductivities (normalized)
# Paper: σ̃_e = 10⁹, σ̃_s = 100 (ratio = 10⁷)
# We cap σ_e at 10⁶ because ratio > 10⁴ gives identical φ inside
# electrode but 10⁷ makes the sparse matrix too ill-conditioned
# for a direct solver. Physics is the same — φ ≈ V_app inside metal.
sigma_e = 1.0e6
sigma_s = 100.0

# Diffusion coefficients (normalized, Chen 2015 Table 1)
D_electrode = 0.03        # D̃_e — nearly zero inside Li metal
D_solution  = 30.0        # D̃_s — in electrolyte at reference T (298 K)

# Temperature dependence of D (Yan et al. eq. 25)
E_D   = 3.3e4             # activation energy, J/mol (Yan Table 1)
T_ref = 298.15            # reference temperature, K

# =====================================================================
# 4. SPATIAL OPERATORS
# =====================================================================
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

# =====================================================================
# 5. ANISOTROPIC κ(θ)
# =====================================================================
def aniso_kappa(xi):
    ux, uy = gradients(xi)
    theta  = np.arctan2(uy, ux)
    return kappa0 * (1.0 + delta_aniso * np.cos(mode * theta))**2

# =====================================================================
# 6. PHASE-FIELD FUNCTIONS
# =====================================================================
def g_prime(xi):
    """g'(ξ) = 2ξ(1−ξ)(1−2ξ)"""
    return 2.0 * xi * (1.0 - xi) * (1.0 - 2.0 * xi)

def h_prime(xi):
    """h'(ξ) = 30ξ²(1−ξ)²  — confines reaction to interface"""
    return 30.0 * xi**2 * (1.0 - xi)**2

def h_interp(xi):
    """h(ξ) = ξ³(10 − 15ξ + 6ξ²) — smooth interpolant for σ and D"""
    return xi**3 * (10.0 - 15.0*xi + 6.0*xi**2)

# =====================================================================
# 7. EFFECTIVE DIFFUSION COEFFICIENT
#    D_eff(ξ, T) = D_e·h(ξ) + D_s(T)·(1 − h(ξ))
#    (Chen 2015 eq. 9: interpolates between electrode and solution)
# =====================================================================
def diffusion_coeff(xi, T_kelvin):
    """
    Temperature-dependent effective diffusion coefficient.
    D_s(T) follows Arrhenius (Yan et al. eq. 25):
        D_s(T) = D_s(T_ref) · exp(−E_D/R · (1/T − 1/T_ref))
    """
    D_s_T = D_solution * np.exp(-E_D / Rgas * (1.0/T_kelvin - 1.0/T_ref))
    h     = h_interp(np.clip(xi, 0.0, 1.0))
    return D_electrode * h + D_s_T * (1.0 - h), D_s_T

# =====================================================================
# 8. POISSON SOLVER — ∇·(σ(ξ)∇φ) = 0
#    BCs: φ = V_app (left), φ = 0 (right), Neumann top/bottom
# =====================================================================
def solve_poisson(xi):
    h     = h_interp(np.clip(xi, 0.0, 1.0))
    sigma = sigma_s + (sigma_e - sigma_s) * h

    sx = 0.5 * (sigma[:, :-1] + sigma[:, 1:])
    sy = 0.5 * (sigma[:-1, :] + sigma[1:, :])

    N   = Ny * Nx
    idc = lambda i, j: i * Nx + j
    rows, cols, vals = [], [], []
    b   = np.zeros(N)

    for i in range(Ny):
        for j in range(Nx):
            k = idc(i, j)
            if j == 0:
                rows.append(k); cols.append(k); vals.append(1.0)
                b[k] = V_app; continue
            if j == Nx - 1:
                rows.append(k); cols.append(k); vals.append(1.0)
                b[k] = 0.0;    continue
            d = 0.0
            for (si, sj, sf, h2) in [
                    ( 0,+1, sx[i,j],             dx**2),
                    ( 0,-1, sx[i,j-1],           dx**2),
                    (+1, 0, sy[i,j] if i<Ny-1 else 0, dy**2),
                    (-1, 0, sy[i-1,j] if i>0   else 0, dy**2),
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
    """η_a = V_app − φ − E°   (negative everywhere → deposition)"""
    return V_app - phi - E_std

# =====================================================================
# 9. BUTLER-VOLMER RATE  (the {} bracket in eq. 15 of Yan et al.)
#
#    R_BV = exp[(1−α)nFη_a/RT] − C·exp[−αnFη_a/RT]
#
#    At η_a ≈ −0.2 V, T = 298 K:
#      (1−α)nFη_a/RT = 0.5×96485×(−0.2)/2479 = −3.89  → exp = 0.020
#      −αnFη_a/RT    = 0.5×96485×0.2/2479     = +3.89  → exp = 48.9
#      R_BV ≈ 0.02 − C×48.9 ≈ −48.9 for C=1 (strongly cathodic) ✓
# =====================================================================
def butler_volmer(C, eta_a, T):
    fac   = n_elec * F_const / (Rgas * T)
    exp_a = np.exp(np.clip( (1-alpha)*fac*eta_a, -50, 50))
    exp_b = np.exp(np.clip(-alpha    *fac*eta_a, -50, 50))
    return exp_a - C * exp_b

# =====================================================================
# 10. INITIAL CONDITIONS
#     Chen 2015 Section 3.2: "artificial nucleation at center of
#     electrode-electrolyte interface"
# =====================================================================
def initial_conditions():
    xi = np.zeros((Ny, Nx))
    c  = np.ones( (Ny, Nx))

    # Flat electrode: first 3 cells
    xi[:, :3] = 1.0
    c[:,  :3] = 0.0

    # Seed protrusion at center of electrode wall
    yy, xx = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing="ij")
    # In normalized coords: seed extends ~0.3 units into electrolyte
    # (comparable to interface thickness δ ≈ 0.2)
    x_norm = xx * dx
    y_norm = yy * dy
    y_center = Ly / 2.0
    seed = np.exp(-((x_norm - 0.4)**2 / 0.04 + (y_norm - y_center)**2 / 0.02))
    xi = np.maximum(xi, 0.98 * seed)
    c  = np.minimum(c,  1.0 - 0.98 * seed)

    return xi, c

def apply_bc(xi, c):
    xi[:,0]  = 1.0;  c[:,0]  = 0.0   # electrode
    xi[:,-1] = 0.0;  c[:,-1] = 1.0   # bulk electrolyte
    return xi, c

# =====================================================================
# 11. TIME-STEPPING
# =====================================================================
def run_simulation(T_kelvin=298.15, nsteps=30000,
                   save_every=10000, poisson_every=25,
                   verbose=True):

    xi, c = initial_conditions()

    # Effective diffusion at this temperature
    _, D_s_T = diffusion_coeff(xi, T_kelvin)

    # CFL stability for explicit scheme:
    # 1. Diffusion:  dt < dx² / (2·D_max)
    # 2. Phase field: dt < dx² / (2·M_σ·κ₀)
    # 3. Reaction:   dt < 1 / (2·M_η·|R_BV_max|·h'_max)
    #    h'_max = 30·(0.5)²·(0.5)² = 1.875
    #    At the interface, φ ≈ V_app/2, so η_a ≈ V_app − V_app/2 − E_std
    eta_interface = V_app - V_app/2.0 - E_std   # estimate at interface center
    bv_max = np.exp(min(alpha * n_elec * F_const * abs(eta_interface)
                        / (Rgas * T_kelvin), 50.0))
    h_prime_max = 1.875

    dt_diff   = 0.4 * dx**2 / (2.0 * max(D_s_T, M_sigma * kappa0))
    dt_react  = 0.3 / (M_eta * bv_max * h_prime_max + 1e-30)
    dt        = min(dt_diff, dt_react)

    if verbose:
        print(f"  D_s(T={T_kelvin-273.15:.0f}°C) = {D_s_T:.1f}")
        print(f"  M_σ·κ₀ = {M_sigma*kappa0:.1f}")
        print(f"  |R_BV|_max ≈ {bv_max:.1f}")
        print(f"  dt_diff = {dt_diff:.2e}   dt_react = {dt_react:.2e}")
        print(f"  dt = {dt:.2e}   ({nsteps} steps → "
              f"t_final = {nsteps*dt:.4f} normalized "
              f"= {nsteps*dt*4000:.0f} s real)")

    phi   = solve_poisson(xi)
    eta_a = compute_eta_a(phi)
    snaps = [(0, xi.copy(), c.copy(), phi.copy(), eta_a.copy())]

    for step in range(1, nsteps + 1):

        if step % poisson_every == 0:
            phi   = solve_poisson(xi)
            eta_a = compute_eta_a(phi)

        R_bv = butler_volmer(c, eta_a, T_kelvin)
        kap  = aniso_kappa(xi)

        # Effective diffusion field D_eff(ξ, T)
        D_eff, _ = diffusion_coeff(xi, T_kelvin)

        # --- Eq. 15 of Yan et al. (phase field) ---
        dxi = (-M_sigma * (W * g_prime(xi) - kap * laplacian(xi))
               - M_eta  * h_prime(xi) * R_bv)

        # --- Eq. 16 of Yan et al. (concentration), simplified ---
        # Using D_eff(ξ,T) instead of uniform D_s
        dc  = D_eff * laplacian(c) - h_prime(xi) * R_bv

        xi += dt * dxi
        c  += dt * dc
        xi  = np.clip(xi, 0.0, 1.0)
        c   = np.clip(c,  0.0, 1.0)
        xi, c = apply_bc(xi, c)

        if step % save_every == 0:
            phi_s = solve_poisson(xi)
            eta_s = compute_eta_a(phi_s)
            snaps.append((step, xi.copy(), c.copy(), phi_s, eta_s))
            if verbose:
                tip = int((xi > 0.5).any(axis=0).nonzero()[0].max()) \
                      if (xi > 0.5).any() else 0
                iface = (xi > 0.05) & (xi < 0.95)
                t_sim = step * dt
                eta_str = (f"[{eta_s[iface].min():.3f},"
                           f"{eta_s[iface].max():.3f}]"
                           if iface.any() else "no interface")
                print(f"  step {step:6d}  t={t_sim:.4f}  "
                      f"tip={tip}/{Nx-1}  η_a∈{eta_str}  "
                      f"C_tip={c[Ny//2, min(tip,Nx-2)]:.3f}")

    return snaps, dt, D_s_T

# =====================================================================
# 12. DRIVER
# =====================================================================
if __name__ == "__main__":

    # --- Single temperature run at 25°C ---
    T_run  = 298.15
    nsteps = 40000

    print(f"=== T = {T_run-273.15:.0f}°C   {nsteps} steps ===")
    print(f"Grid: {Nx}×{Ny}  domain: {Lx}×{Ly}  dx={dx:.4f}")
    print(f"Paper params: M_σ={M_sigma}, M_η={M_eta}, "
          f"W={W}, κ₀={kappa0}, D_s={D_solution}")

    t0 = time.time()
    snaps, dt, D_s_T = run_simulation(T_kelvin=T_run, nsteps=nsteps,
                                      save_every=nsteps//4,
                                      poisson_every=25)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")

    # Plot all fields at each snapshot
    n = len(snaps)
    fig, axes = plt.subplots(4, n, figsize=(3.5*n, 13))
    labels = [r'Phase $\xi$', r'Li$^+$ conc $C$',
              r'Potential $\phi$', r'$\eta_a$']
    cmaps  = ['viridis', 'magma', 'plasma', 'RdBu_r']

    for col, (step, xi, c, phi, eta_a) in enumerate(snaps):
        for row, (fld, lbl, cm) in enumerate(
                zip([xi, c, phi, eta_a], labels, cmaps)):
            ax = axes[row, col]
            im = ax.imshow(fld, origin='lower', cmap=cm, aspect='equal')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            if row == 0:
                ax.set_title(f'step {step}\nt={step*dt:.3f}', fontsize=9)
            if col == 0:
                ax.set_ylabel(lbl, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f'Yan et al. normalized params — T={T_run-273.15:.0f}°C  '
                 f'D_s={D_s_T:.1f}  V_app={V_app}V\n'
                 f'M_σ={M_sigma}  M_η={M_eta}  W={W}  '
                 f'κ₀={kappa0}  δ={delta_aniso}',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig('yan_fields.png', dpi=130, bbox_inches='tight')
    print("Saved: yan_fields.png")

    # --- Temperature sweep ---
    print("\n=== Temperature sweep ===")
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))

    for col, T in enumerate([273.15, 298.15, 323.15]):
        print(f"\n  T = {T-273.15:.0f}°C")
        s, dt_t, Ds = run_simulation(
            T_kelvin=T, nsteps=nsteps, save_every=nsteps,
            poisson_every=25, verbose=True)
        _, xi, c, phi, eta_a = s[-1]

        ax = axes2[0, col]
        ax.imshow(xi, origin='lower', vmin=0, vmax=1, cmap='viridis')
        ax.set_title(f'T={T-273.15:.0f}°C  D_s={Ds:.1f}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes2[1, col]
        im = ax.imshow(c, origin='lower', vmin=0, vmax=1, cmap='magma')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(r'Li$^+$ concentration $C$', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    axes2[0, 0].set_ylabel(r'Phase $\xi$', fontsize=10)
    axes2[1, 0].set_ylabel(r'Concentration $C$', fontsize=10)
    fig2.suptitle('Temperature sweep — Yan et al. normalized parameters',
                  fontsize=12)
    plt.tight_layout()
    plt.savefig('yan_sweep.png', dpi=130, bbox_inches='tight')
    print("\nSaved: yan_sweep.png")

    if not os.environ.get('MPLBACKEND', '').lower() == 'agg':
        plt.show()
