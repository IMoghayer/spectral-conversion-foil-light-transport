import numpy as np
import matplotlib.pyplot as plt
from pytmatrix import tmatrix  # ← one import line is enough
from tqdm import tqdm
from itertools import product

#scatterer setup
# Restrict to only forward incidence for polar plot test
theta_in_deg = np.linspace(0, 180, 2)
phi_in_deg = np.linspace(0, 360, 2, endpoint=False)

theta_out_deg = np.linspace(0, 180, 10)
phi_out_deg = np.linspace(0, 360, 2, endpoint=False)

dθ = theta_out_deg[1] - theta_out_deg[0]
dφ = phi_out_deg[1] - phi_out_deg[0]

import pandas as pd

wavelengths = np.arange(300, 801, 1)
m_values = [1.3, 1.5, 1.7, 2]
all_rows = []

for m_rel in m_values:
    for wl in wavelengths:
        print(f"Processing m = {m_rel}, wavelength {wl} nm...")
        scatterer = tmatrix.Scatterer(
            radius=500.0,
            wavelength=wl,
            m= m_rel + 0.0j,
            axis_ratio=500.0 / 100.0,
            shape=tmatrix.Scatterer.SHAPE_SPHEROID,
            radius_type=tmatrix.Scatterer.RADIUS_MAXIMUM
        )
        scatterer.thet0 = 0.0
        scatterer.thet0 = 0.0
        scatterer.orient = tmatrix.orientation.orient_single

        scatterer.ndgs = 5
        scatterer.ddelt = 1e-2

        rows = []
        total_pairs = list(product(theta_in_deg, phi_in_deg))
        for θ0, φ0 in tqdm(total_pairs, desc=f"Incoming directions for m={m_rel}, {wl}nm"):
            scatterer.thet0 = θ0
            scatterer.phi0 = φ0

            # Loop over outgoing directions without overwriting incoming
            for θdeg in theta_out_deg:
                scatterer.thet = θdeg
                theta_rad = np.deg2rad(θdeg)
                for φdeg in phi_out_deg:
                    scatterer.phi = φdeg

                    Z = scatterer.get_Z()
                    I_val = float(np.real(Z[0, 0]))

                    dOmega = np.sin(theta_rad) * np.deg2rad(dθ) * np.deg2rad(dφ)

                    rows.append([m_rel, wl, θ0, φ0, θdeg, φdeg, I_val, dOmega])

        df = pd.DataFrame(
            rows,
            columns=["m", "wavelength", "theta_in", "phi_in", "theta_out", "phi_out", "I", "dOmega"]
        )

        group_cols = ["m", "wavelength", "theta_in", "phi_in"]

        I_int = df.groupby(group_cols).apply(lambda g: (g["I"] * g["dOmega"]).sum()).rename("I_int")
        df = df.join(I_int, on=group_cols)

        df["phase_func"] = df["I"] / df["I_int"]
        all_rows.append(df)

final_df = pd.concat(all_rows, ignore_index=True)
final_df.to_csv("scattering_direction_pdf_all_wavelengths_Z11_solidangle_m.csv", index=False)
print("Saved full 2D scattering PDF (Z11, solid-angle normalized) for both m to scattering_direction_pdf_all_wavelengths_Z11_solidangle_m.csv")

# Verify normalization: ∑ phase_func * dΩ = 1 for each incoming direction
check_norm = final_df.groupby(["m", "wavelength", "theta_in", "phi_in"]).apply(
    lambda g: float((g["phase_func"] * g["dOmega"]).sum())
)
print("All group integrals ≈ 1:", np.allclose(check_norm.values, 1.0))

# Compute forward fraction and anisotropy g for each wavelength and incoming direction
def compute_forward_fraction_and_g(group):
    # Forward fraction: integral over θ_out ∈ [0, 90°] divided by integral over [0, 180°]
    forward_mask = group["theta_out"] <= 90
    forward_integral = (group.loc[forward_mask, "phase_func"] * group.loc[forward_mask, "dOmega"]).sum()
    total_integral = (group["phase_func"] * group["dOmega"]).sum()
    forward_fraction = forward_integral / total_integral

    # Anisotropy g: average cosθ weighted by phase function over full sphere
    theta_rad = np.deg2rad(group["theta_out"])
    g = (np.cos(theta_rad) * group["phase_func"] * group["dOmega"]).sum() / total_integral

    return pd.Series({"forward_fraction": forward_fraction, "g": g})


fg_df = final_df.groupby(["m", "wavelength", "theta_in", "phi_in"]).apply(compute_forward_fraction_and_g).reset_index()

# Print results grouped by m and wavelength
for m_rel in m_values:
    print(f"\nResults for m = {m_rel}:")
    sub_m = fg_df[fg_df["m"] == m_rel]
    for wl in wavelengths:
        sub = sub_m[sub_m["wavelength"] == wl]
        for _, row in sub.iterrows():
            print(f"  λ={wl} nm, incoming (θ_in={row['theta_in']}, φ_in={row['phi_in']}): F = {row['forward_fraction']:.4f}, g = {row['g']:.4f}")

# --- Helper: combine direct + scattered to get net forward-transmitted fraction ---
def transmitted_fraction_from_F(F, tau):
    """Net forward-going fraction through a slab of optical depth tau.
    tau = n_sigma_L (wavelength-dependent optical depth). If tau is small,
    T ≈ (1 - tau) + tau*F. Exact single-pass Beer+single-scatter expression:
    T = exp(-tau) + (1 - exp(-tau)) * F.
    """
    return np.exp(-tau) + (1.0 - np.exp(-tau)) * F

# Choose a representative optical depth (edit as needed or map vs wavelength)
# tau = n * σ * L, where n = number density, σ = cross section (may depend on λ), L = layer thickness.
# If you have wavelength-dependent tau(λ), you can supply an array here for more realistic curves.
TAU = 0.2  # example single-pass optical depth of the scattering layer


# --- Plots: Forward fraction and anisotropy vs wavelength (θ_in=0, φ_in=0) ---
sel_theta_in = 0.0
sel_phi_in = 0.0
# Use a smooth line with a finer wavelength grid for forward fraction plot
wavelengths = np.linspace(300, 800, 500)
plt.figure(figsize=(7,4))
for m_rel in m_values:
    sub = fg_df[(fg_df["m"] == m_rel) & (fg_df["theta_in"] == sel_theta_in) & (fg_df["phi_in"] == sel_phi_in)]
    plt.plot(sub["wavelength"], sub["forward_fraction"], label=f"n={m_rel}")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Forward fraction (0–90°)")
plt.title("Forward fraction vs wavelength (θ_in=0°, φ_in=0°)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.figure(figsize=(7,4))
for m_rel in m_values:
    sub = fg_df[(fg_df["m"] == m_rel) & (fg_df["theta_in"] == sel_theta_in) & (fg_df["phi_in"] == sel_phi_in)]
    plt.plot(sub["wavelength"], sub["g"], marker='o', label=f"m={m_rel}")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Anisotropy g")
plt.title("Anisotropy vs wavelength (θ_in=0°, φ_in=0°)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# --- Plot: Net forward-transmitted fraction vs wavelength for each m ---
plt.figure(figsize=(7,4))
for m_rel in m_values:
    sub = fg_df[(fg_df["m"] == m_rel) & (fg_df["theta_in"] == sel_theta_in) & (fg_df["phi_in"] == sel_phi_in)]
    T_net = transmitted_fraction_from_F(sub["forward_fraction"].values, TAU)
    plt.plot(sub["wavelength"], T_net, marker='o', label=f"m={m_rel}")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Net forward-transmitted fraction")
plt.title(f"Net transmission vs wavelength (θ_in=0°, φ_in=0°), τ={TAU}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# --- Polar plot from CSV for a specific combination ---
df_loaded = pd.read_csv("scattering_direction_pdf_all_wavelengths_Z11_solidangle_m.csv")

# Choose parameters
check_m = m_values[0]
check_wl = 700
check_theta_in = theta_in_deg[0]
check_phi_in = phi_in_deg[0]
check_phi_out = phi_out_deg[0]

# Filter for a single (m, wavelength, theta_in, phi_in, phi_out)
df_check = df_loaded[
    (df_loaded["m"] == check_m) &
    (df_loaded["wavelength"] == check_wl) &
    (df_loaded["theta_in"] == check_theta_in) &
    (df_loaded["phi_in"] == check_phi_in) &
    (df_loaded["phi_out"] == check_phi_out)
].sort_values("theta_out")

# Compute forward/backward fractions for this φ_out slice (normalized within the slice)
forward_mask_slice = df_check["theta_out"] <= 90
slice_total = (df_check["phase_func"] * df_check["dOmega"]).sum()
slice_forward = (df_check.loc[forward_mask_slice, "phase_func"] * df_check.loc[forward_mask_slice, "dOmega"]).sum()
F_slice = float(slice_forward / slice_total)
B_slice = 1.0 - F_slice
print(f"Slice forward fraction (phi_out={check_phi_out}°): F_slice={F_slice:.4f}, B_slice={B_slice:.4f}")

# Pull the full-sphere single-scatter forward fraction for the same (m, wavelength, theta_in, phi_in)
F_full = float(
    fg_df[
        (fg_df["m"] == check_m) &
        (fg_df["wavelength"] == check_wl) &
        (fg_df["theta_in"] == check_theta_in) &
        (fg_df["phi_in"] == check_phi_in)
    ]["forward_fraction"].iloc[0]
)
print(f"Full-sphere forward fraction for this case: F_full={F_full:.4f}, B_full={1-F_full:.4f}")

# Build closed polar data
theta_full_csv = np.concatenate([df_check["theta_out"].values, 360.0 - df_check["theta_out"].values[::-1]])
intensity_full_csv = np.concatenate([df_check["phase_func"].values, df_check["phase_func"].values[::-1]])
theta_rad_full_csv = np.radians(theta_full_csv)

# Plot
fig_csv = plt.figure(figsize=(6, 6))
ax_csv = fig_csv.add_subplot(111, polar=True)
ax_csv.plot(theta_rad_full_csv, intensity_full_csv, linestyle='-')
ax_csv.set_theta_zero_location("E")
ax_csv.set_theta_direction(-1)
ax_csv.set_title(f"Polar Plot (Z11, solid-angle normalized)\nλ={check_wl} nm, 100 nm thick")
ax_csv.grid(True)
ax_csv.set_yticklabels([])
#ax_csv.set_yscale('log')
# Add small textbox with both fractions
textstr = f"Slice F={F_slice:.2f}, B={B_slice:.2f}\nFull F={F_full:.2f}, B={1-F_full:.2f}"
ax_csv.text(0.02, 0.02, textstr, transform=ax_csv.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', alpha=0.2))
plt.tight_layout()
plt.show()

# --- Direct phase function plot for forward/backscattering ---
# Use the first incoming direction and the first wavelength
first_theta_in = theta_in_deg[0]
first_phi_in = phi_in_deg[0]
phi_out_val = phi_out_deg[0]
first_wavelength = wavelengths[0]
'''
# Recreate scatterer for phase function
scatterer = tmatrix.Scatterer(
    radius=500.0,
    wavelength=300.0,
    m=1.3 + 0.0j,
    axis_ratio=500.0 / 1000.0,
    shape=tmatrix.Scatterer.SHAPE_SPHEROID,
    radius_type=tmatrix.Scatterer.RADIUS_MAXIMUM
)
scatterer.orient = tmatrix.orientation.orient_single
scatterer.ndgs = 5
scatterer.ddelt = 1e-2
scatterer.thet0 = first_theta_in
scatterer.phi0 = first_phi_in

# Compute intensity I(θ) for fixed φ_out
I_list = []
for θdeg in theta_out_deg:
    scatterer.thet = θdeg
    scatterer.phi = phi_out_val
    S1, S2 = scatterer.get_S()
    I_val = (
        np.abs(S1[0])**2 + np.abs(S1[1])**2 +
        np.abs(S2[0])**2 + np.abs(S2[1])**2
    )
    I_list.append(I_val)

# Plot phase function vs scattering angle
plt.figure(figsize=(8, 4))
plt.plot(theta_out_deg, I_list, marker='o')
plt.xlabel("Scattering angle θ_out (deg)")
plt.ylabel(f"Intensity at φ_out = {phi_out_val}°")
plt.title(f"Phase Function at φ_out = {phi_out_val}°, λ={first_wavelength} nm")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Polar plot of phase function ---
# Build a closed 0–360° polar curve by mirroring 0–180°
theta_full = np.concatenate([theta_out_deg, 360.0 - theta_out_deg[::-1]])
I_full = np.concatenate([I_list, I_list[::-1]])
theta_rad_full = np.radians(theta_full)

# Create polar plot
fig2 = plt.figure(figsize=(6, 6))
ax2 = fig2.add_subplot(111, polar=True)
ax2.plot(theta_rad_full, I_full, marker='o', linestyle='-')
# Set 0° at right (East) and clockwise direction
ax2.set_theta_zero_location("E")
ax2.set_theta_direction(-1)
ax2.set_title(
    f"Polar Phase Function at φ_out = {phi_out_val}°, λ={first_wavelength} nm",
    va='bottom'
)
ax2.grid(True)
plt.tight_layout()
plt.show()
'''