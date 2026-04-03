"""
Ergun Equation Parametric Study — Packed Bed (CO2/N2 Flue Gas)
================================================================
Geometry: Cylindrical extrudate (not sphere)
  - Equivalent diameter: d_eq = (3/2 * de^2 * le)^(1/3)
  - Sphericity: phi = pi*d_eq^2 / A_surface_cylinder

Ergun Equation:
  dP/L = 150*mu*u0*(1-eps)^2 / (phi^2 * d_eq^2 * eps^3)
       + 1.75*rho*u0^2*(1-eps) / (phi * d_eq * eps^3)

Units: SI throughout (Pa, m, kg/m3, Pa.s, m/s)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent  # save output next to this script

# ============================================================
# GLOBAL PARAMETERS — edit here
# ============================================================

# --- Fluid: CO2/N2 flue gas ---
COMPOSITION_CO2 = 0.15          # mol fraction CO2
COMPOSITION_N2  = 0.85          # mol fraction N2

T_C  = 50.0                     # Temperature [C]
T_K  = T_C + 273.15             # Temperature [K]
P_Pa = 3.0 * 1e5                # Pressure [Pa]  (3 bara)

# Viscosity @ 50C (pressure negligible for gas viscosity)
MU_CO2 = 1.61e-5                # Pa.s
MU_N2  = 1.96e-5                # Pa.s
MU = COMPOSITION_CO2 * MU_CO2 + COMPOSITION_N2 * MU_N2

# Density — ideal gas
M_CO2 = 0.04401                 # kg/mol
M_N2  = 0.02801                 # kg/mol
M_MIX = COMPOSITION_CO2 * M_CO2 + COMPOSITION_N2 * M_N2
R     = 8.314
RHO   = (P_Pa * M_MIX) / (R * T_K)

# --- Packing ---
EPS = 0.37                      # void fraction (typical random extrudate packing)

# --- Operating condition ---
U0_CM_MIN = 248.81              # superficial velocity [cm/min]
U0 = U0_CM_MIN / 100.0 / 60.0  # [m/s]

# ============================================================
# PARAMETRIC RANGES — edit here
# ============================================================

# Case 1: Vary Bed Diameter D  (fix L, de, le)
D_RANGE     = np.array([0.01, 0.016, 0.02, 0.03, 0.04, 0.05,
                         0.06, 0.07, 0.08, 0.09, 0.10, 0.20,
                         0.30, 0.40, 0.50])
FIXED_L_C1  = 1.0
FIXED_DE_C1 = 3e-3
FIXED_LE_C1 = 3e-3

# Case 2: Vary Bed Length L  (fix D, de, le)
L_RANGE     = np.round(np.arange(0.1, 2.1, 0.1), 2)
FIXED_D_C2  = 0.05
FIXED_DE_C2 = 3e-3
FIXED_LE_C2 = 3e-3

# Case 3a: Vary extrudate diameter de  (fix L, D, le)
DE_RANGE     = np.array([0.001, 0.0015, 0.002, 0.003, 0.004,
                          0.005, 0.006, 0.007, 0.008, 0.009,
                          0.010, 0.011, 0.012, 0.015, 0.020])
FIXED_L_C3A  = 1.0
FIXED_D_C3A  = 0.05
FIXED_LE_C3A = 5e-3

# Case 3b: Vary extrudate length le  (fix L, D, de)
LE_RANGE     = np.array([0.001, 0.002, 0.003, 0.004, 0.005,
                          0.006, 0.007, 0.008, 0.009, 0.010,
                          0.011, 0.012, 0.015, 0.020])
FIXED_L_C3B  = 1.0
FIXED_D_C3B  = 0.05
FIXED_DE_C3B = 3e-3

# ============================================================
# EXTRUDATE GEOMETRY
# ============================================================

def extrudate_deq(de, le):
    """Volume-equivalent sphere diameter [m]."""
    return (1.5 * de**2 * le) ** (1.0/3.0)

def extrudate_phi(de, le):
    """Sphericity of cylindrical extrudate."""
    d_eq  = extrudate_deq(de, le)
    A_sph = np.pi * d_eq**2
    A_cyl = 0.5 * np.pi * de**2 + np.pi * de * le
    return A_sph / A_cyl

# ============================================================
# CORE ERGUN
# ============================================================

def ergun_dP(L, d_eq, phi, u0=U0, mu=MU, rho=RHO, eps=EPS):
    """Total pressure drop [Pa]."""
    dP_dL = (
        150 * mu * u0 * (1 - eps)**2 / (phi**2 * d_eq**2 * eps**3)
        + 1.75 * rho * u0**2 * (1 - eps) / (phi * d_eq * eps**3)
    )
    return dP_dL * L

def reynolds_mod(d_eq, phi, u0=U0, mu=MU, rho=RHO, eps=EPS):
    return rho * u0 * phi * d_eq / (mu * (1 - eps))

def flow_regime(Re):
    if Re < 10:   return "viscous"
    if Re > 1000: return "turbulent"
    return "transition"

def ergun_extrudate(L, de, le, **kw):
    """ΔP from extrudate dimensions directly."""
    return ergun_dP(L, extrudate_deq(de, le), extrudate_phi(de, le), **kw)

# ============================================================
# COMPUTE ALL CASES
# ============================================================

dP_C1  = ergun_extrudate(FIXED_L_C1, FIXED_DE_C1, FIXED_LE_C1)
dP_C2  = np.array([ergun_extrudate(L, FIXED_DE_C2, FIXED_LE_C2) for L in L_RANGE])
dP_C3A = np.array([ergun_extrudate(FIXED_L_C3A, de, FIXED_LE_C3A) for de in DE_RANGE])
dP_C3B = np.array([ergun_extrudate(FIXED_L_C3B, FIXED_DE_C3B, le) for le in LE_RANGE])

# ============================================================
# PLOTTING
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"Ergun Equation — CO2/N2 Flue Gas  |  Cylindrical Extrudate Packing\n"
    f"T={T_C:.0f} C,  P={P_Pa/1e5:.0f} bara,  "
    f"u0={U0_CM_MIN:.2f} cm/min ({U0*1000:.2f} mm/s),  "
    f"rho={RHO:.3f} kg/m3,  mu={MU:.3e} Pa.s,  eps={EPS}",
    fontsize=10, fontweight='bold'
)

# Case 1
ax1 = axes[0, 0]
d_eq_c1 = extrudate_deq(FIXED_DE_C1, FIXED_LE_C1)
phi_c1  = extrudate_phi(FIXED_DE_C1, FIXED_LE_C1)
Re_c1   = reynolds_mod(d_eq_c1, phi_c1)
ax1.axhline(dP_C1 / 1000, color='steelblue', lw=2.5,
            label=f"dP = {dP_C1:.1f} Pa (constant)")
ax1.set_xlim([D_RANGE.min(), D_RANGE.max()])
ax1.set_ylim([0, dP_C1 / 1000 * 2.5])
ax1.set_xlabel("Bed Diameter D [m]", fontsize=11)
ax1.set_ylabel("dP [kPa]", fontsize=11)
ax1.set_title(
    f"Case 1: Vary D  (L={FIXED_L_C1}m, de={FIXED_DE_C1*1000:.1f}mm, le={FIXED_LE_C1*1000:.1f}mm)",
    fontsize=10)
ax1.text(0.5, 0.55,
    f"dP is independent of D\n(Ergun = pressure drop per unit cross-section area)\n\n"
    f"d_eq = {d_eq_c1*1000:.2f} mm,  phi = {phi_c1:.3f}\nRe_mod = {Re_c1:.1f} ({flow_regime(Re_c1)})",
    transform=ax1.transAxes, ha='center', va='center', fontsize=9, color='gray',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Case 2
ax2 = axes[0, 1]
ax2.plot(L_RANGE, dP_C2 / 1000, 'o-', color='darkorange', lw=2, ms=5)
for idx in [0, 4, 9, -1]:
    ax2.annotate(f"{dP_C2[idx]:.0f} Pa",
                 (L_RANGE[idx], dP_C2[idx]/1000),
                 textcoords="offset points", xytext=(6, 4), fontsize=8)
ax2.set_xlabel("Bed Length L [m]", fontsize=11)
ax2.set_ylabel("dP [kPa]", fontsize=11)
ax2.set_title(
    f"Case 2: Vary L  (D={FIXED_D_C2}m, de={FIXED_DE_C2*1000:.1f}mm, le={FIXED_LE_C2*1000:.1f}mm)",
    fontsize=10)
ax2.grid(True, alpha=0.3)

# Case 3a
ax3 = axes[1, 0]
ax3.plot(DE_RANGE * 1000, dP_C3A / 1000, 's-', color='crimson', lw=2, ms=5)
for idx in [0, 4, 9, -1]:
    ax3.annotate(f"{dP_C3A[idx]:.0f} Pa",
                 (DE_RANGE[idx]*1000, dP_C3A[idx]/1000),
                 textcoords="offset points", xytext=(4, 4), fontsize=8)
ax3.set_xlabel("Extrudate Diameter de [mm]", fontsize=11)
ax3.set_ylabel("dP [kPa]", fontsize=11)
ax3.set_title(
    f"Case 3a: Vary de  (L={FIXED_L_C3A}m, D={FIXED_D_C3A}m, le={FIXED_LE_C3A*1000:.1f}mm fixed)",
    fontsize=10)
ax3.grid(True, alpha=0.3)

# Case 3b
ax4 = axes[1, 1]
ax4.plot(LE_RANGE * 1000, dP_C3B / 1000, 'D-', color='darkgreen', lw=2, ms=5)
for idx in [0, 4, 9, -1]:
    ax4.annotate(f"{dP_C3B[idx]:.0f} Pa",
                 (LE_RANGE[idx]*1000, dP_C3B[idx]/1000),
                 textcoords="offset points", xytext=(4, 4), fontsize=8)
ax4.set_xlabel("Extrudate Length le [mm]", fontsize=11)
ax4.set_ylabel("dP [kPa]", fontsize=11)
ax4.set_title(
    f"Case 3b: Vary le  (L={FIXED_L_C3B}m, D={FIXED_D_C3B}m, de={FIXED_DE_C3B*1000:.1f}mm fixed)",
    fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / "ergun_study.png", dpi=150, bbox_inches='tight')

# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("=" * 65)
print("  FLUID PROPERTIES")
print("=" * 65)
print(f"  T         = {T_C} C  ({T_K:.2f} K)")
print(f"  P         = {P_Pa/1e5:.1f} bara")
print(f"  u0        = {U0_CM_MIN} cm/min  =  {U0:.5f} m/s")
print(f"  rho       = {RHO:.4f} kg/m3")
print(f"  mu        = {MU:.4e} Pa.s")
print(f"  M_mix     = {M_MIX*1000:.2f} g/mol")
print("=" * 65)

print("\n  Case 2: dP vs L")
print(f"  {'L [m]':>7}  {'dP [Pa]':>10}  {'dP [kPa]':>10}")
for L, dP in zip(L_RANGE, dP_C2):
    print(f"  {L:>7.1f}  {dP:>10.1f}  {dP/1000:>10.3f}")

print("\n  Case 3a: dP vs de  (le fixed)")
print(f"  {'de[mm]':>7}  {'d_eq[mm]':>9}  {'phi':>6}  {'dP [Pa]':>10}")
for de, dP in zip(DE_RANGE, dP_C3A):
    deq = extrudate_deq(de, FIXED_LE_C3A)
    phi = extrudate_phi(de, FIXED_LE_C3A)
    print(f"  {de*1000:>7.2f}  {deq*1000:>9.3f}  {phi:>6.3f}  {dP:>10.1f}")

print("\n  Case 3b: dP vs le  (de fixed)")
print(f"  {'le[mm]':>7}  {'d_eq[mm]':>9}  {'phi':>6}  {'dP [Pa]':>10}")
for le, dP in zip(LE_RANGE, dP_C3B):
    deq = extrudate_deq(FIXED_DE_C3B, le)
    phi = extrudate_phi(FIXED_DE_C3B, le)
    print(f"  {le*1000:>7.2f}  {deq*1000:>9.3f}  {phi:>6.3f}  {dP:>10.1f}")

print(f"\nPlot saved -> {SCRIPT_DIR / 'ergun_study.png'}")

# ============================================================
# FUNCTION INTERFACE
# Usage: python ergun_study.py L de le [D] [u0_cm_min]
# ============================================================

def query(L, de, le, D=None, u0=U0, mu=MU, rho=RHO, eps=EPS):
    d_eq = extrudate_deq(de, le)
    phi  = extrudate_phi(de, le)
    dP   = ergun_dP(L, d_eq, phi, u0=u0, mu=mu, rho=rho, eps=eps)
    Re   = reynolds_mod(d_eq, phi, u0=u0, mu=mu, rho=rho, eps=eps)
    print("\n" + "=" * 55)
    print("  ERGUN QUERY RESULT")
    print("=" * 55)
    print(f"  L      = {L} m")
    print(f"  de     = {de*1000:.3f} mm")
    print(f"  le     = {le*1000:.3f} mm")
    print(f"  d_eq   = {d_eq*1000:.3f} mm  (volume-equiv. sphere)")
    print(f"  phi    = {phi:.4f}")
    if D: print(f"  D      = {D} m  (no effect on dP)")
    print(f"  u0     = {u0*100*60:.2f} cm/min")
    print(f"  rho    = {rho:.4f} kg/m3")
    print(f"  mu     = {mu:.4e} Pa.s")
    print(f"  eps    = {eps}")
    print("-" * 55)
    print(f"  Re_mod = {Re:.2f}  -> {flow_regime(Re)}")
    print(f"  dP     = {dP:.2f} Pa")
    print(f"         = {dP/1000:.4f} kPa")
    print(f"         = {dP/1e5:.6f} bar")
    print(f"  dP/L   = {dP/L:.2f} Pa/m")
    print("=" * 55 + "\n")
    return dP

if __name__ == "__main__" and len(sys.argv) > 1:
    args = sys.argv[1:]
    if args[0] in ("--help", "-h"):
        print("\nUsage: python ergun_study.py L de le [D] [u0_cm_min]")
        print("  L         = bed length [m]")
        print("  de        = extrudate diameter [m]")
        print("  le        = extrudate length [m]")
        print("  D         = bed diameter [m]  (optional)")
        print("  u0_cm_min = superficial velocity [cm/min]  (optional)")
        print("\nExample:")
        print("  python ergun_study.py 1.0 0.003 0.003")
        print("  python ergun_study.py 2.0 0.002 0.005 0.05 200")
    else:
        L_in  = float(args[0])
        de_in = float(args[1])
        le_in = float(args[2])
        D_in  = float(args[3]) if len(args) > 3 else None
        u0_in = float(args[4]) / 100.0 / 60.0 if len(args) > 4 else U0
        query(L_in, de_in, le_in, D=D_in, u0=u0_in)