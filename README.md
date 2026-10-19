# PhaseFieldModel
Phase Field Model


The goal is to replicate, step by step, the model in Yan et al. (2018), A computational investigation of thermal effect on lithium dendrite growth. That paper extends the nonlinear phase-field model of Chen et al. (2015) with a heat-transfer model. Both papers solved their models in COMSOL with implicit finite elements; this repository rebuilds the model with explicit finite differences in NumPy/SciPy.

**Running:**
Requires Python 3 with NumPy, SciPy, and Matplotlib
"python -m pip install numpy scipy matplotlib"

**li_dendrite_v3.py:**
The script runs one 40,000-step simulation at 25 °C and saves yan_fields.png, then runs three more for the temperature sweep and saves yan_sweep.png, then opens the results in a plot window. Each run takes roughly 10-15 minutes. Progress is printed every 10,000 steps. To save the images without opening windows, set the environment variable MPLBACKEND=Agg before running.

***Model*** \
**Units/Domain:** \
All quantities are normalized as in Chen et al. (2015): length l₀ = 100 μm, energy density E₀ = 1.5×10⁶ J/m³, time Δt₀ = 4000 s. The domain is 5 × 5 normalized units (500 × 500 μm) on a 150 × 150 uniform grid. The electrode is the left wall and the bulk electrolyte is the right wall.
ξ: 	Phase field: 1 = Li metal, 0 = electrolyte
C: 	Li⁺ concentration, normalized by the bulk concentration c₀
φ: Electric potential (V)
η_a: Activation overpotential (V)
T: Temperature (K), one uniform value per run

**Equations used in V3** \
Phase field (Yan eq. 15, Chen eq. 8): \
    ∂ξ/∂t = −M_σ [ W g′(ξ) − κ(θ) ∇²ξ ] − M_η h′(ξ) R_BV

Butler–Volmer rate: \
    R_BV = exp[(1−α) nF η_a / RT] − C · exp[−α nF η_a / RT]

Activation overpotential (Yan eq. 10, with E° = 0): \
    η_a = φ − E°

Li⁺ concentration (simplified from Yan eq. 18 / Chen eq. 9): \
    ∂C/∂t = D_eff(ξ,T) ∇²C − (c_s/c₀) ∂ξ/∂t

Electric potential (Yan eq. 19 / Chen eq. 10, without the source term): \
    ∇·[ σ(ξ) ∇φ ] = 0

Temperature-dependent diffusivity (Yan eq. 25, without the concentration factor): \
    D_s(T) = D_s(T_ref) · exp[ −(E_D/R)(1/T − 1/T_ref) ]

Function names follow Chen's notation: g_prime is the double-well derivative, 2ξ(1−ξ)(1−2ξ), and h_prime is the derivative of the interpolation function h(ξ) = ξ³(10 − 15ξ + 6ξ²). Yan et al. use h for the double well and L for the interpolation function. The effective diffusivity and conductivity are interpolated between electrode and electrolyte values with h(ξ). The anisotropy is κ(θ) = κ₀[1 + δ cos(4θ)]².

**Boundary and initial conditions** \
Left wall (electrode): 	ξ=1; C=0; φ:V_app = −0.2 V \
Right wall (bulk electrolyte): 	ξ=0; C=1; φ:V_app = 0 \ 
Top and bottom: No Flux

**Parameters compared with the papers** \
Interfacial mobility M_σ: Paper=2000; v3=7.5; Explicit stability; both mobilities divided by ~267, keeping M_η/M_σ = 2 \
Reaction constant M_η: Paper=4000; v3=15; same as above \
Barrier height W: 25 \
Gradient coefficient κ₀: .01 \
Anisotropy strength δ: .05 \
D in electrode: .03 \
D in electrolyte (298 K): 30 \
Conductivity, electrode: Paper=10^9; v3=10^6; Keeps the sparse direct solve well conditioned \
Conductivity, electrolyte: 100 \
Asymmetry factor α: .5 \
c_s/c₀: 76.4 \
Activation energy E_D: 3.3×10⁴ J/mol \
Applied voltage: Paper=−0.45 to −3.0 V (Chen 2D cases); v3=−0.2 V; The stable explicit time step shrinks roughly exponentially with |V_app| \
Grid Spacing: Paper=2 μm minimum, adaptive (Chen); v3=33.6 μm, uniform \
Time Step: Paper: 0.2 s, implicit; v3: 2.85×10⁻⁶ (≈ 0.011 s), explicit

**History** \
1. allen_cahn.py: 2D Allen–Cahn test. \
2. OneDimensionalDendriteSmoothing.py: 1D diffuse interface. \
3. li_dendrite_pf.py (not in this repository): phase field and concentration with a fixed overpotential. \
4. li_dendrite_poisson.py: adds the Poisson solve for a spatially varying overpotential. \
6. li_dendrite_v3.py: paper parameters.

**References** \
L. Chen, H.W. Zhang, L.Y. Liang, Z. Liu, Y. Qi, P. Lu, J. Chen, L.-Q. Chen. Modulation of dendritic patterns during electrodeposition: A nonlinear phase-field model. Journal of Power Sources 300 (2015) 376–385. https://doi.org/10.1016/j.jpowsour.2015.09.055

H.H. Yan, Y.H. Bie, X.Y. Cui, G.P. Xiong, L. Chen. A computational investigation of thermal effect on lithium dendrite growth. Energy Conversion and Management (2018). https://www.sciencedirect.com/science/article/pii/S0196890418300980
