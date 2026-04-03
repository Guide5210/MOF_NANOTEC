"""
Ergun Equation — Interactive Parametric Study
==============================================
Cylindrical extrudate packing, CO2/N2 flue gas
Run: python ergun_study.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# ============================================================
# GLOBAL DEFAULTS — edit here if needed
# ============================================================
COMPOSITION_CO2 = 0.15
COMPOSITION_N2  = 0.85
T_C             = 50.0
T_K             = T_C + 273.15
P_Pa            = 3.0e5
MU_CO2          = 1.61e-5
MU_N2           = 1.96e-5
MU              = COMPOSITION_CO2 * MU_CO2 + COMPOSITION_N2 * MU_N2
M_CO2           = 0.04401
M_N2            = 0.02801
M_MIX           = COMPOSITION_CO2 * M_CO2 + COMPOSITION_N2 * M_N2
R               = 8.314
RHO             = (P_Pa * M_MIX) / (R * T_K)
EPS             = 0.37
U0_CM_MIN       = 248.81
U0              = U0_CM_MIN / 100.0 / 60.0

# ============================================================
# GEOMETRY & PHYSICS
# ============================================================

def extrudate_deq(de, le):
    return (1.5 * de**2 * le) ** (1.0 / 3.0)

def extrudate_phi(de, le):
    d_eq  = extrudate_deq(de, le)
    A_sph = np.pi * d_eq**2
    A_cyl = 0.5 * np.pi * de**2 + np.pi * de * le
    return A_sph / A_cyl

def ergun_dP(L, d_eq, phi, u0=U0, mu=MU, rho=RHO, eps=EPS):
    dP_dL = (
        150 * mu * u0 * (1 - eps)**2 / (phi**2 * d_eq**2 * eps**3)
        + 1.75 * rho * u0**2 * (1 - eps) / (phi * d_eq * eps**3)
    )
    return dP_dL * L

def reynolds_mod(d_eq, phi, u0=U0, mu=MU, rho=RHO, eps=EPS):
    return rho * u0 * phi * d_eq / (mu * (1 - eps))

def flow_regime(Re):
    if Re < 10:   return "viscous (Blake-Kozeny)"
    if Re > 1000: return "turbulent (Burke-Plummer)"
    return "transition"

def ergun_extrudate(L, de, le, **kw):
    return ergun_dP(L, extrudate_deq(de, le), extrudate_phi(de, le), **kw)

# ============================================================
# INPUT HELPERS
# ============================================================

def ask(prompt, default=None, unit=""):
    suffix = f" [{default} {unit}]" if default is not None else f" [{unit}]" if unit else ""
    while True:
        raw = input(f"  {prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("  !! กรุณาใส่ตัวเลข")

def ask_range(param_name, unit):
    print(f"\n  กำหนด range ของ {param_name} [{unit}]")
    start = ask("  start", unit=unit)
    stop  = ask("  stop",  unit=unit)
    n     = int(ask("  จำนวนจุด", default=10))
    return np.linspace(start, stop, n)

def sep(char="─", n=55):
    print(char * n)

# ============================================================
# PLOT HELPER
# ============================================================

def save_plot(fig, filename):
    out = SCRIPT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Plot saved → {out}")

# ============================================================
# MODES
# ============================================================

def mode_single():
    """กรอกค่าตรงๆ ได้ dP ออกมา"""
    sep("═")
    print("  MODE: Single Query — กรอกค่าทุกตัว → ได้ dP")
    sep("═")
    L  = ask("Bed length L", unit="m")
    de = ask("Extrudate diameter de", unit="m")
    le = ask("Extrudate length le",   unit="m")
    u0_input = ask("Superficial velocity u0", default=U0_CM_MIN, unit="cm/min")
    u0 = u0_input / 100.0 / 60.0

    d_eq = extrudate_deq(de, le)
    phi  = extrudate_phi(de, le)
    dP   = ergun_dP(L, d_eq, phi, u0=u0)
    Re   = reynolds_mod(d_eq, phi, u0=u0)

    sep()
    print("  RESULT")
    sep()
    print(f"  d_eq   = {d_eq*1000:.3f} mm")
    print(f"  phi    = {phi:.4f}")
    print(f"  Re_mod = {Re:.2f}  → {flow_regime(Re)}")
    print(f"  dP     = {dP:.2f} Pa  =  {dP/1000:.4f} kPa  =  {dP/1e5:.6f} bar")
    print(f"  dP/L   = {dP/L:.2f} Pa/m")
    sep()


def mode_vary_L():
    """Vary L, fix D, de, le"""
    sep("═")
    print("  MODE: Vary Bed Length L  (fix D, de, le)")
    sep("═")
    D  = ask("Bed diameter D (fix)", unit="m")
    de = ask("Extrudate diameter de (fix)", unit="m")
    le = ask("Extrudate length le (fix)",   unit="m")
    u0_input = ask("Superficial velocity u0", default=U0_CM_MIN, unit="cm/min")
    u0 = u0_input / 100.0 / 60.0
    L_arr = ask_range("L", "m")

    dP_arr = np.array([ergun_extrudate(L, de, le, u0=u0) for L in L_arr])
    d_eq = extrudate_deq(de, le)
    phi  = extrudate_phi(de, le)
    Re   = reynolds_mod(d_eq, phi, u0=u0)

    print(f"\n  {'L [m]':>8}  {'dP [Pa]':>10}  {'dP [kPa]':>10}")
    sep("-", 35)
    for L, dP in zip(L_arr, dP_arr):
        print(f"  {L:>8.3f}  {dP:>10.1f}  {dP/1000:>10.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(L_arr, dP_arr / 1000, 'o-', color='darkorange', lw=2, ms=6)
    ax.set_xlabel("Bed Length L [m]", fontsize=12)
    ax.set_ylabel("ΔP [kPa]", fontsize=12)
    ax.set_title(
        f"Ergun: ΔP vs Bed Length L\n"
        f"D={D*100:.1f}cm, de={de*1000:.2f}mm, le={le*1000:.2f}mm, "
        f"u₀={u0_input:.1f}cm/min, Re={Re:.1f} ({flow_regime(Re)})",
        fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_plot(fig, "ergun_vary_L.png")
    plt.show()


def mode_vary_D():
    """Vary D — explain independence, show info"""
    sep("═")
    print("  MODE: Vary Bed Diameter D  (fix L, de, le)")
    sep("═")
    print("  ⚠  หมายเหตุ: Ergun equation คำนวณ dP/unit cross-section area")
    print("     ดังนั้น ΔP ไม่ขึ้นกับ D เลย ถ้า u0 คงที่")
    print("     Mode นี้จะแสดง ΔP คงที่ + แสดง Re, phi เพื่อ reference\n")

    L  = ask("Bed length L (fix)", unit="m")
    de = ask("Extrudate diameter de (fix)", unit="m")
    le = ask("Extrudate length le (fix)",   unit="m")
    u0_input = ask("Superficial velocity u0", default=U0_CM_MIN, unit="cm/min")
    u0   = u0_input / 100.0 / 60.0
    D_arr = ask_range("D", "m")

    dP    = ergun_extrudate(L, de, le, u0=u0)
    d_eq  = extrudate_deq(de, le)
    phi   = extrudate_phi(de, le)
    Re    = reynolds_mod(d_eq, phi, u0=u0)

    sep()
    print(f"  ΔP = {dP:.2f} Pa  =  {dP/1000:.4f} kPa  (คงที่ทุก D)")
    print(f"  d_eq = {d_eq*1000:.3f} mm,  phi = {phi:.4f},  Re = {Re:.2f} ({flow_regime(Re)})")
    sep()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(dP / 1000, color='steelblue', lw=2.5,
               label=f"ΔP = {dP:.1f} Pa (constant — independent of D)")
    ax.set_xlim([D_arr.min(), D_arr.max()])
    ax.set_ylim([0, dP / 1000 * 2.5])
    ax.set_xlabel("Bed Diameter D [m]", fontsize=12)
    ax.set_ylabel("ΔP [kPa]", fontsize=12)
    ax.set_title(
        f"Ergun: ΔP vs Bed Diameter D\n"
        f"L={L}m, de={de*1000:.2f}mm, le={le*1000:.2f}mm  |  "
        f"ΔP independent of D when u₀ is fixed",
        fontsize=10)
    ax.text(0.5, 0.6,
        f"d_eq = {d_eq*1000:.2f} mm\nφ = {phi:.4f}\n"
        f"Re_mod = {Re:.1f}\n({flow_regime(Re)})",
        transform=ax.transAxes, ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_plot(fig, "ergun_vary_D.png")
    plt.show()


def mode_vary_de():
    """Vary extrudate diameter de, fix L, D, le"""
    sep("═")
    print("  MODE: Vary Extrudate Diameter de  (fix L, D, le)")
    sep("═")
    L  = ask("Bed length L (fix)", unit="m")
    D  = ask("Bed diameter D (fix)", unit="m")
    le = ask("Extrudate length le (fix)", unit="m")
    u0_input = ask("Superficial velocity u0", default=U0_CM_MIN, unit="cm/min")
    u0 = u0_input / 100.0 / 60.0
    de_arr = ask_range("de", "m")

    dP_arr  = np.array([ergun_extrudate(L, de, le, u0=u0) for de in de_arr])
    deq_arr = np.array([extrudate_deq(de, le) for de in de_arr])
    phi_arr = np.array([extrudate_phi(de, le) for de in de_arr])

    print(f"\n  {'de[mm]':>8}  {'d_eq[mm]':>9}  {'phi':>6}  {'dP [Pa]':>10}  {'dP [kPa]':>10}")
    sep("-", 52)
    for de, deq, phi, dP in zip(de_arr, deq_arr, phi_arr, dP_arr):
        print(f"  {de*1000:>8.3f}  {deq*1000:>9.3f}  {phi:>6.3f}  {dP:>10.1f}  {dP/1000:>10.4f}")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(de_arr * 1000, dP_arr / 1000, 's-', color='crimson', lw=2, ms=6, label="ΔP")
    ax1.set_xlabel("Extrudate Diameter de [mm]", fontsize=12)
    ax1.set_ylabel("ΔP [kPa]", fontsize=12, color='crimson')
    ax1.tick_params(axis='y', labelcolor='crimson')

    ax2 = ax1.twinx()
    ax2.plot(de_arr * 1000, phi_arr, '--', color='navy', lw=1.5, ms=5, label="φ (sphericity)")
    ax2.set_ylabel("Sphericity φ [-]", fontsize=12, color='navy')
    ax2.tick_params(axis='y', labelcolor='navy')

    ax1.set_title(
        f"Ergun: ΔP vs Extrudate Diameter de\n"
        f"L={L}m, D={D*100:.1f}cm, le={le*1000:.2f}mm, u₀={u0_input:.1f}cm/min",
        fontsize=10)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    save_plot(fig, "ergun_vary_de.png")
    plt.show()


def mode_vary_le():
    """Vary extrudate length le, fix L, D, de"""
    sep("═")
    print("  MODE: Vary Extrudate Length le  (fix L, D, de)")
    sep("═")
    L  = ask("Bed length L (fix)", unit="m")
    D  = ask("Bed diameter D (fix)", unit="m")
    de = ask("Extrudate diameter de (fix)", unit="m")
    u0_input = ask("Superficial velocity u0", default=U0_CM_MIN, unit="cm/min")
    u0 = u0_input / 100.0 / 60.0
    le_arr = ask_range("le", "m")

    dP_arr  = np.array([ergun_extrudate(L, de, le, u0=u0) for le in le_arr])
    deq_arr = np.array([extrudate_deq(de, le) for le in le_arr])
    phi_arr = np.array([extrudate_phi(de, le) for le in le_arr])

    print(f"\n  {'le[mm]':>8}  {'d_eq[mm]':>9}  {'phi':>6}  {'dP [Pa]':>10}  {'dP [kPa]':>10}")
    sep("-", 52)
    for le, deq, phi, dP in zip(le_arr, deq_arr, phi_arr, dP_arr):
        print(f"  {le*1000:>8.3f}  {deq*1000:>9.3f}  {phi:>6.3f}  {dP:>10.1f}  {dP/1000:>10.4f}")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(le_arr * 1000, dP_arr / 1000, 'D-', color='darkgreen', lw=2, ms=6, label="ΔP")
    ax1.set_xlabel("Extrudate Length le [mm]", fontsize=12)
    ax1.set_ylabel("ΔP [kPa]", fontsize=12, color='darkgreen')
    ax1.tick_params(axis='y', labelcolor='darkgreen')

    ax2 = ax1.twinx()
    ax2.plot(le_arr * 1000, phi_arr, '--', color='navy', lw=1.5, label="φ (sphericity)")
    ax2.set_ylabel("Sphericity φ [-]", fontsize=12, color='navy')
    ax2.tick_params(axis='y', labelcolor='navy')

    ax1.set_title(
        f"Ergun: ΔP vs Extrudate Length le\n"
        f"L={L}m, D={D*100:.1f}cm, de={de*1000:.2f}mm, u₀={u0_input:.1f}cm/min",
        fontsize=10)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    save_plot(fig, "ergun_vary_le.png")
    plt.show()


# ============================================================
# MAIN MENU
# ============================================================

MODES = {
    "1": ("Single query — กรอกค่าทุกตัว ได้ dP เดียว",    mode_single),
    "2": ("Vary Bed Length L       (fix D, de, le)",        mode_vary_L),
    "3": ("Vary Bed Diameter D     (fix L, de, le)",        mode_vary_D),
    "4": ("Vary Extrudate de       (fix L, D, le)",         mode_vary_de),
    "5": ("Vary Extrudate le       (fix L, D, de)",         mode_vary_le),
}

def print_fluid():
    sep("═")
    print("  FLUID: CO2/N2 Flue Gas")
    sep()
    print(f"  T={T_C}°C  P={P_Pa/1e5:.0f} bara  u0={U0_CM_MIN}cm/min")
    print(f"  rho={RHO:.4f} kg/m3  mu={MU:.4e} Pa.s  eps={EPS}")
    sep("═")

def main():
    print("\n" + "═"*55)
    print("  ERGUN EQUATION — Interactive Parametric Study")
    print("  Cylindrical Extrudate Packing | CO2/N2 Flue Gas")
    print("═"*55)
    print_fluid()

    while True:
        print("\n  เลือก Mode:")
        for k, (desc, _) in MODES.items():
            print(f"    [{k}]  {desc}")
        print("    [q]  ออก")

        choice = input("\n  > ").strip().lower()

        if choice == "q":
            print("\n  Goodbye.\n")
            break
        elif choice in MODES:
            try:
                MODES[choice][1]()
            except KeyboardInterrupt:
                print("\n  (ยกเลิก — กลับเมนู)")
        else:
            print("  !! กรุณาเลือก 1-5 หรือ q")

        again = input("\n  ต้องการรัน mode อื่นอีกมั้ย? (y/n) [y]: ").strip().lower()
        if again == "n":
            print("\n  Goodbye.\n")
            break

if __name__ == "__main__":
    main()
