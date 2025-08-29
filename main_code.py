import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm
from tqdm import tqdm
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')

# turn this to False to completely disable scattering
ENABLE_SCATTERING = False

# Load McCree quantum yield 
mccree_df = pd.read_csv("/home/ilia/Desktop/mccree_table4_CO2_yield.csv")

# Simulation parameters
N_rays = 100000
surface_size = (1e4, 1e4)  # surface at which light is incident
theta_deg = 0.0  # vertical incidence
phi_deg = 0.0
quantum_yield = 0.9
frac = 0.0005
concentration = 0.02 / frac

from collections import defaultdict
# Load scattering direction PDF for all wavelengths
scatter_df = pd.read_csv("/home/ilia/Desktop/scattering_direction_pdf_all_wavelengths5.csv")

# --- Quick polar plot check from CSV ---
check_wl = scatter_df['wavelength'].iloc[0]
check_theta_in = scatter_df['theta_in'].iloc[0]
check_phi_in = scatter_df['phi_in'].iloc[0]
check_phi_out = scatter_df['phi_out'].iloc[0]

df_check = scatter_df[
    (scatter_df['wavelength'] == check_wl) &
    (scatter_df['theta_in'] == check_theta_in) &
    (scatter_df['phi_in'] == check_phi_in) &
    (scatter_df['phi_out'] == check_phi_out)
].sort_values('theta_out')

theta_full_csv = np.concatenate([df_check['theta_out'].values, 360.0 - df_check['theta_out'].values[::-1]])
prob_full_csv = np.concatenate([df_check['probability'].values, df_check['probability'].values[::-1]])
theta_rad_full_csv = np.radians(theta_full_csv)

# Load emission and absorption data (replace with actual paths if needed)
abs_file = "/home/ilia/Downloads/ExportAbsorption_JV87.Sample.Raw.asc.csv"
em_file = "/home/ilia/Downloads/ExportPL_JV91_Exc390_Cen550_NewM266Gr1_Filter4_t5000ms_StartTime20220909145841.txt.csv"
em_file = '/home/ilia/Downloads/ExportPL_JV90_Exc390_Cen550_NewM266Gr1_Filter4_t5000ms_StartTime20220909145841.txt.csv'

#abs_file = '/home/ilia/Desktop/perovskite_green_OD.txt'

# Read and rename columns
abs_df = pd.read_csv(abs_file, header=None, names=["wavelength_nm", "absorption"])
em_df = pd.read_csv(em_file, header=None, names=["wavelength_nm", "emission"])

def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Fit Gaussian to emission data
popt, _ = curve_fit(gaussian, em_df["wavelength_nm"], em_df["emission"],
                    p0=[1, em_df["wavelength_nm"][em_df["emission"].idxmax()], 20])

a, mu, sigma = popt
fine_wavelengths = np.linspace(300, 800, 10000)  

em_pdf_vals = gaussian(fine_wavelengths, a, mu, sigma)
em_pdf_vals /= np.sum(em_pdf_vals)

emission_pdf = interp1d(fine_wavelengths, em_pdf_vals, bounds_error=False, fill_value=0)

# Define a new emission wavelength grid and PDF for emission sampling
em_wavelengths = np.linspace(300, 800, 10000)
em_pdf_vals = gaussian(em_wavelengths, a, mu, sigma)
em_pdf_vals /= np.sum(em_pdf_vals)
emission_pdf = interp1d(em_wavelengths, em_pdf_vals, bounds_error=False, fill_value=0)

abs_wavelengths = abs_df["wavelength_nm"].values
abs_probs = abs_df["absorption"].values
absorption_pdf_raw = interp1d(abs_wavelengths, abs_probs, bounds_error=False, fill_value=0)



# === Define Scattering and Absorption PDF based on power law fit ===

# Use transformed absorption: -ln(1 - absorption)
transmission_wl = abs_df["wavelength_nm"].values
ln_absorption_alt = -np.log(np.clip(1 - abs_df["absorption"], 1e-12, 1.0))

# --- Compute and plot kappa from 1 - I/I0 spectrum ---
# Assumptions:
#   • Absorption column is (1 - T) measured in a 1 cm cuvette.
#   • We scale to full concentration by 'scale_factor' (e.g., 2000×).
path_length_m = 0.01            # 1 cm cuvette
scale_factor = 2000             # concentration scaling to full-strength material

wavelength_nm_arr = abs_df["wavelength_nm"].values
wavelength_m = wavelength_nm_arr * 1e-9

# Convert (1 - T) → μ_meas using Beer–Lambert:
# μ_meas = -ln(T) / L = -ln(1 - absorption) / L
mu_meas = ln_absorption_alt / max(path_length_m, 1e-12)   # m^-1, measured dilution
mu_full = mu_meas * scale_factor                           # m^-1, full concentration

# κ from μ: μ = 4πκ/λ  ⇒ κ = μλ/(4π)
kappa = mu_full * wavelength_m / (4 * np.pi)

# Save κ(λ) to CSV
kappa_df = pd.DataFrame({
    "wavelength_nm": wavelength_nm_arr,
    "kappa": kappa
})
kappa_df.to_csv("kappa_spectrum.csv", index=False)

# Optional: print κ at a reference wavelength if in range (e.g., 320 nm)
try:
    ref_nm = 320.0
    if (wavelength_nm_arr.min() <= ref_nm) and (ref_nm <= wavelength_nm_arr.max()):
        # nearest index
        idx_ref = int(np.argmin(np.abs(wavelength_nm_arr - ref_nm)))
        print(f"κ({wavelength_nm_arr[idx_ref]:.1f} nm) ≈ {kappa[idx_ref]:.3e}")
except Exception as _e:
    pass

# Restrict to 600-800 nm for fitting
fit_mask = (transmission_wl >= 650) & (transmission_wl <= 800)
x_fit = transmission_wl[fit_mask]
y_fit = ln_absorption_alt[fit_mask]

# Define power law
def power_law(wavelength, a, b):
    return a * wavelength**b

# Fit power law
popt, _ = curve_fit(power_law, x_fit, y_fit, p0=[4e3, 1.0])
a_fit, b_fit = popt

print(b_fit)

# Define the scattering PDF from the power law fit
def scattering_pdf(wavelengths):
    vals = power_law(np.asarray(wavelengths), a_fit, b_fit)
    # Ensure scattering probability in [0,1]
    return np.clip(vals, 0.0, 1e6)

# Define the absorption PDF as the residual between absorbance and scattering
def absorption_pdf(wavelengths):
    wavelengths = np.asarray(wavelengths)
    interp_abs = interp1d(transmission_wl, ln_absorption_alt, bounds_error=False, fill_value=0)
    raw_vals = interp_abs(wavelengths)
    scatter_vals = scattering_pdf(wavelengths)
    residual = raw_vals - scatter_vals
    residual[residual < 0] = 0
    return residual



def mfp(wavelengths):
    mu_values = absorption_pdf(wavelengths)
    wavelengths = np.asarray(wavelengths)
    # Zero out mu_values for wavelengths > 450 nm
    mu_values = np.where(wavelengths > 450, 0.0, mu_values)
    alpha = np.clip(mu_values, 1e-10, 1e10)
    rho = np.random.uniform(0, 1, size=mu_values.shape)
    log_term = np.log(1 / np.clip(rho, 1e-12, 1.0))
    log_term = np.clip(log_term, 0, 100)
    mfp_values = (1e7 / alpha) * log_term * frac
    mfp_values[mfp_values < 0] = 1e15
    return mfp_values

def mfp_scattering(wavelengths):
    mu_s_values = scattering_pdf(wavelengths)
    alpha = np.clip(mu_s_values, 1e-10, 1e10)
    rho = np.random.uniform(0, 1, size=mu_s_values.shape)
    log_term = np.log(1 / np.clip(rho, 1e-12, 1.0))
    log_term = np.clip(log_term, 0, 100)
    mfp_values = (1e7 / alpha) * log_term * frac
    mfp_values[mfp_values < 0] = 1e15
    return mfp_values



# New function to sample emission wavelengths vectorized for array of cutoffs
def sample_emission_wavelength_vectorized(absorbed_wavelengths):
    absorbed_wavelengths = np.asarray(absorbed_wavelengths)
    out = np.empty_like(absorbed_wavelengths)

    for i, lam_abs in enumerate(absorbed_wavelengths):
        mask = em_wavelengths >= lam_abs
        if not np.any(mask):
            out[i] = lam_abs
            continue
        choices = em_wavelengths[mask]
        probs = em_pdf_vals[mask]
        probs /= probs.sum()
        out[i] = np.random.choice(choices, p=probs)
    return out

# Load AM1.5 spectrum
data = np.genfromtxt("/home/ilia/Downloads/ASTMG173.csv", delimiter=",", skip_header=2)

sun_wavelengths = data[:, 0]        # Wavelengths in nm
sun_irradiance = data[:, 2]         # Global tilt W/m�/nm

# Restrict sun spectrum to 300-1000 nm for sampling
valid_range = (sun_wavelengths >= 300) & (sun_wavelengths <= 800)
sun_wavelengths = sun_wavelengths[valid_range]
sun_irradiance = sun_irradiance[valid_range]

photon_flux = sun_irradiance * sun_wavelengths
prob_density = photon_flux / np.sum(photon_flux)

# Sample wavelengths based on restricted AM1.5 spectrum
# Interpolate to a finer wavelength resolution

sun_interp = interp1d(sun_wavelengths, prob_density, kind='linear', bounds_error=False, fill_value=0)
fine_wavelengths = np.linspace(300, 800, 10000)
fine_probs = sun_interp(fine_wavelengths)
fine_probs /= np.sum(fine_probs)

sampled_wavelengths = np.random.choice(fine_wavelengths, size=N_rays, p=fine_probs)


# Round all sampled wavelengths to the nearest integer
#sampled_wavelengths = np.round(sampled_wavelengths).astype(int)

# Sample random positions on the surface
x = np.random.uniform(0, surface_size[0], N_rays)
y = np.random.uniform(0, surface_size[1], N_rays)

mfp_absorption = mfp(sampled_wavelengths)
mfp_scat = mfp_scattering(sampled_wavelengths)
# Define array
dtype = np.dtype([
    ('x', 'f8'),
    ('y', 'f8'),
    ('z', 'f8'),
    ('wavelength_nm', 'f8'),
    ('initial_wavelength_nm', 'f8'),  # <-- ADD THIS LINE
    ('theta_deg', 'f8'),
    ('phi_deg', 'f8'),
    ('alive', 'bool'),
    ('transmitted', 'bool'),
    ('mfp_absorption', 'f8'),
    ('mfp_scattering', 'f8'),
    ('initial', 'bool'),
    ('reflected', 'bool'),
    ('path_traversed', 'f8'),  # total distance traveled in nm
    ('absorbed',  'bool'),
])


# Initialize rays
rays = np.zeros(N_rays, dtype=dtype)
rays['reflected'] = False
rays['absorbed']  = False
rays['path_traversed'] = 0.0

rays['x'] = x
rays['y'] = y
rays['z'] = 0.0
rays['wavelength_nm'] = sampled_wavelengths
rays['initial_wavelength_nm'] = sampled_wavelengths.copy()  # Store true initial values

rays['theta_deg'] = np.degrees(np.arccos(np.sqrt(np.random.uniform(0, 1, N_rays))))
rays['phi_deg'] = np.random.uniform(0, 360, N_rays)

rays['theta_deg'] = 0.0  # incoming from directly above
rays['phi_deg'] = 0.0    # azimuthal angle doesn't matter if theta=0


rays['alive'] = True
rays['transmitted'] = False
rays['mfp_absorption'] = mfp_absorption
rays['mfp_scattering'] = mfp_scat
rays['initial'] = True


# Initial Fresnel reflection at air-material interface
theta_i_rad = np.radians(rays['theta_deg'])
n1 = 1.0  # air
n2 = 1.5  # material

sin_theta_t = n1 / n2 * np.sin(theta_i_rad)
tir_mask = np.abs(sin_theta_t) > 1
theta_t_rad = np.where(tir_mask, 0, np.arcsin(np.clip(sin_theta_t, -1, 1)))

Rs = ((n1 * np.cos(theta_i_rad) - n2 * np.cos(theta_t_rad)) /
      (n1 * np.cos(theta_i_rad) + n2 * np.cos(theta_t_rad)))**2
Rp = ((n1 * np.cos(theta_t_rad) - n2 * np.cos(theta_i_rad)) /
      (n1 * np.cos(theta_t_rad) + n2 * np.cos(theta_i_rad)))**2
R_entry = 0.5 * (Rs + Rp)

reflect_entry = np.random.rand(len(rays)) < R_entry

# Rays that reflect never enter
rays['alive'][reflect_entry] = False
rays['transmitted'][reflect_entry] = False  # Optional: mark explicitly
print(f"Initial Fresnel reflection: {np.mean(reflect_entry):.4f} of rays reflected at entry.")
# instead of just killing them, also flag
rays['reflected'][reflect_entry] = True

total_absorptions = 0
total_scatters = 0

# === Plot -ln(1 - Absorption) and Power Law Fit with μ on Y-axis ===
# Fit power law to μ_s from abs_df, using only 700–800 nm for the fit
def power_law(wavelength, a, b):
    return a * wavelength**b

wavelengths_fit = abs_df["wavelength_nm"].values
mu_s_fit = -np.log(np.clip(1 - abs_df["absorption"].values, 1e-12, 1.0))
# Restrict fitting to 700–800 nm
fit_mask = (wavelengths_fit >= 650) & (wavelengths_fit <= 800)
popt, _ = curve_fit(power_law, wavelengths_fit[fit_mask], mu_s_fit[fit_mask], p0=[1e-5, -1.0])
a_fit, b_fit = popt
mu_fit = power_law(wavelengths_fit, a_fit, b_fit)

from scipy.spatial import KDTree

# Precompute a lookup dictionary for scattering directions for fast access
scatter_lookup = defaultdict(list)
for _, row in tqdm(scatter_df.iterrows(), total=len(scatter_df), desc="Building scatter lookup"):
    key = (int(row['wavelength']), int(row['theta_in']), int(row['phi_in']))
    scatter_lookup[key].append((row['theta_out'], row['phi_out'], row['probability']))

available_keys_list = list(scatter_lookup.keys())
available_keys_array = np.array(available_keys_list)
key_tree = KDTree(available_keys_array)

# Precompute keys for fast nearest-neighbor lookup
lookup_keys_array = np.array(list(scatter_lookup.keys()))  # shape: (N, 3)

# ── Precompute for super‐fast lookups ──
# scatter_lookup: dict[(wl, θ_in, φ_in)] → list of (θ_out, φ_out, prob)

# 1) all available wavelengths
wl_keys = np.array(sorted({key[0] for key in scatter_lookup.keys()}))

# 2) for each wavelength, list of available (θ_in, φ_in)
from collections import defaultdict
in_angles = defaultdict(list)
for wl, θi, φi in scatter_lookup:
    in_angles[wl].append((θi, φi))

total_scatters = 0

# === Plot -ln(1 - Absorption) and Power Law Fit with μ on Y-axis ===
plt.figure(figsize=(10, 5))

# Compute μ_s from absorption data
ln_absorption = -np.log(np.clip(1 - abs_df["absorption"], 1e-12, 1.0))
mu_s = 1e-7 * ln_absorption

# Evaluate power-law fit over the same wavelength range
fit_wavelengths = abs_df["wavelength_nm"]
mu_fit = power_law(fit_wavelengths, a_fit, b_fit)

# === Plot absorption_pdf and scattering_pdf functions (with dual y-axis) ===
plot_wavelengths = np.linspace(300, 800, 1000)
abs_vals = absorption_pdf(plot_wavelengths) / 1e-2 * 1/frac
scat_vals = scattering_pdf(plot_wavelengths) / 1e-2 * 1/frac

# κ from μ: μ = 4πκ/λ  ⇒ κ = μλ/(4π)
kappa = absorption_pdf(wavelength_nm_arr)/1e7 * wavelength_m / (4 * np.pi)

# Save κ(λ) to CSV
kappa_df = pd.DataFrame({
    "wavelength_nm": wavelength_nm_arr,
    "kappa": kappa
})
kappa_df.to_csv("kappa_spectrum.csv", index=False)


# Time evolution loop
#max_steps = 100000
step = 0
with tqdm(total=N_rays, desc="Simulating rays", dynamic_ncols=False, ncols=80) as pbar:
    while np.sum(rays['alive']) > 0.002 * N_rays:
        alive = rays['alive'] 
        step += 1
        #print(f"Step {step}: {np.sum(rays['alive']) / N_rays:.3f} fraction of rays still alive")
        #print(f"Step {step}: {np.sum(rays['transmitted']) / N_rays:.3f} fraction of rays transmitted")
        step_size = 1000
        theta_rad = np.radians(rays['theta_deg'])
        phi_rad = np.radians(rays['phi_deg'])

        dx = step_size * np.sin(theta_rad) * np.cos(phi_rad)
        dy = step_size * np.sin(theta_rad) * np.sin(phi_rad)
        dz = step_size * np.cos(theta_rad)

        # Check if next z-position would leave the Perovskite
        z_next = rays['z'] + dz
        exiting_top = (z_next < 0) & alive
        exiting_bottom = (z_next > 100000) & alive
        # Side exit checks (foil is 10 mm x 10 mm)
        exiting_x_low = rays['x'] + dx < 0
        exiting_x_high = rays['x'] + dx > 10e30
        exiting_y_low = rays['y'] + dy < 0
        exiting_y_high = rays['y'] + dy > 10e30
        side_reflect_x = exiting_x_low | exiting_x_high
        side_reflect_y = exiting_y_low | exiting_y_high
        exiting = exiting_top | exiting_bottom
        still_inside = (~exiting) & alive    # 👈 now only rays that are still alive

        # Fresnel equations (angle-dependent, per-ray, including TIR handling)
        n1 = 1.5  # Refractive index of perovskite
        n2 = 1.0  # Refractive index of air

        theta_i_rad = np.radians(rays['theta_deg'])
        # Use Snell's law to compute theta_t, guarding against total internal reflection
        sin_theta_t = n1 / n2 * np.sin(theta_i_rad)
        # Total internal reflection where sin(theta_t) > 1
        tir_mask = np.abs(sin_theta_t) > 1
        theta_t_rad = np.where(tir_mask, 0, np.arcsin(np.clip(sin_theta_t, -1, 1)))

        # Fresnel reflectance for s- and p-polarized light
        Rs = ((n1 * np.cos(theta_i_rad) - n2 * np.cos(theta_t_rad)) / (n1 * np.cos(theta_i_rad) + n2 * np.cos(theta_t_rad)))**2
        Rp = ((n1 * np.cos(theta_t_rad) - n2 * np.cos(theta_i_rad)) / (n1 * np.cos(theta_t_rad) + n2 * np.cos(theta_i_rad)))**2

        # Average for unpolarized light
        R = 0.5 * (Rs + Rp)
        R = np.clip(R, 0, 1)  # ensure numerical safety
        T = 1 - R

        # Transmit or reflect light particles leaving the Perovskite according to Fresnel
        transmission_test = (np.random.rand(len(rays)) < T) & alive

        transmit_top = exiting_top & transmission_test
        transmit_bottom = exiting_bottom & transmission_test

        #print(f"Step {step}: Fraction transmitted this step = {np.mean(transmit_top | transmit_bottom):.4f}")

        # Mark transmitted rays as transmitted and not alive
        rays['transmitted'][transmit_bottom] = True
        rays['alive'][transmit_top | transmit_bottom] = False

        exit_reflect = exiting_top & ~transmission_test
        internal_bounce = exiting_bottom & ~transmission_test

        # final reflection out the top
        rays['reflected'][exit_reflect] = True
        rays['alive'][exit_reflect]   = False

        # for a “bounce” off the bottom face, just flip direction
        rays['theta_deg'][internal_bounce] = 180.0 - rays['theta_deg'][internal_bounce]



        # Reflect rays at the sides by inverting the corresponding direction angle
        rays['phi_deg'][side_reflect_x] = 180.0 - rays['phi_deg'][side_reflect_x]
        rays['phi_deg'][side_reflect_y] *= -1

        # Update position for remaining alive rays
        rays['x'][still_inside] += dx[still_inside]
        rays['y'][still_inside] += dy[still_inside]
        rays['z'][still_inside] += dz[still_inside]

        # === Recalculate who is still inside after movement ===
        still_inside = (
            (rays['z'] >= 0) & (rays['z'] <= 100000) &
            (rays['x'] >= 0) & (rays['x'] <= 10e30) &
            (rays['y'] >= 0) & (rays['y'] <= 10e30) &
            (rays['alive'])  # must still be alive
        )


        # Update scattering mean free path for each ray
        rays['mfp_scattering'][still_inside] -= step_size
        # Update absorption mean free path for each ray
        safe_mask = still_inside & (rays['mfp_absorption'] < 1e29)
        rays['mfp_absorption'][safe_mask] -= step_size


                # Find rays that should scatter
        # Find rays that should scatter (but zero‐out if scattering is disabled)
        if ENABLE_SCATTERING:
            to_scatter = still_inside & (rays['mfp_scattering'] <= 0)
        else:
            to_scatter = np.zeros_like(still_inside, dtype=bool)

        # count how many scattered this step
        n_scatter = to_scatter.sum()
        total_scatters += n_scatter

        # only do the work if scattering is enabled and there are any
        if ENABLE_SCATTERING and n_scatter > 0:
            scatter_indices = np.where(to_scatter)[0]
            # … (your existing code that samples θ_in, φ_in; looks up candidates; builds new_directions; assigns rays["theta_deg"] and ["phi_deg"]) …

            # reset their mfp_scattering
            rays['mfp_scattering'][to_scatter] = mfp_scattering(rays['wavelength_nm'][to_scatter])

        n_scatter = np.sum(to_scatter)
        total_scatters += n_scatter
        # Find rays that should be absorbed
        to_absorb = still_inside & (rays['mfp_absorption'] <= 0)

        # Quantum yield logic for absorption event
        rand_vals = np.random.rand(len(rays))
        emit_mask = to_absorb & (rand_vals < quantum_yield)
        terminate_mask = to_absorb & (rand_vals >= quantum_yield)
        n_abs = np.sum(to_absorb)
        n_emit = np.sum(emit_mask)
        if n_abs > 0:
            print(f"💡 Absorbed this step: {n_abs}, Re-emitted: {n_emit} ({n_emit/n_abs:.2%})")

        # Handle reemission: assign new random direction and wavelength
        rays['wavelength_nm'][emit_mask] = sample_emission_wavelength_vectorized(rays['wavelength_nm'][emit_mask])
        rays['theta_deg'][emit_mask] = np.degrees(np.arccos(np.random.uniform(-1, 1, size=np.sum(emit_mask))))
        rays['phi_deg'][emit_mask] = np.random.uniform(0, 360, size=np.sum(emit_mask))
        rays['mfp_absorption'][emit_mask] = mfp(rays['wavelength_nm'][emit_mask])
        rays['initial'][emit_mask] = False


        # Terminate rays not reemitted
        rays['absorbed'][terminate_mask] = True
        rays['alive'][terminate_mask]    = False

        # Handle scattering event
        scattering_event_mask = to_scatter
        scatter_indices = np.where(scattering_event_mask)[0]
        num_rays = len(scatter_indices)

        if np.any(to_scatter):
            idxs = np.where(to_scatter)[0]
            new_dirs = np.zeros((len(idxs), 3))

            for j, idx in enumerate(idxs):
                wl = rays['wavelength_nm'][idx]
                # 1) find nearest wavelength in table
                wl_key = int(wl_keys[np.argmin(np.abs(wl_keys - wl))])

                # 2) use the actual incoming ray direction to pick the nearest table orientation
                ray_theta = rays['theta_deg'][idx]
                ray_phi = rays['phi_deg'][idx] % 360
                cands = in_angles[wl_key]
                min_dist = float('inf')
                theta_i_sel = None
                phi_i_sel = None
                for ti, pi in cands:
                    # compute minimal angular difference
                    d_theta = abs(ray_theta - ti)
                    d_phi = abs((ray_phi - pi + 180) % 360 - 180)
                    d = d_theta + d_phi
                    if d < min_dist:
                        min_dist = d
                        theta_i_sel = ti
                        phi_i_sel = pi
                θi, φi = theta_i_sel, phi_i_sel

                # 3) sample (θ_out, φ_out) from that entry’s distribution
                key = (wl_key, θi, φi)
                candidates = scatter_lookup[key]
                θo_list, φo_list, p_list = zip(*candidates)
                θo = np.radians(np.random.choice(θo_list, p=p_list))
                φo = np.radians(np.random.choice(φo_list, p=p_list))

                # 4) build local frame around current ray direction
                t0 = np.radians(rays['theta_deg'][idx])
                p0 = np.radians(rays['phi_deg'][idx])
                z = np.array([np.sin(t0)*np.cos(p0),
                            np.sin(t0)*np.sin(p0),
                            np.cos(t0)])
                # pick any x ⟂ z
                if np.allclose(z, [0,0,1]):
                    x = np.array([1,0,0])
                else:
                    x = np.cross([0,0,1], z)
                    x /= np.linalg.norm(x)
                y = np.cross(z, x)

                # rotate by (θo, φo) in that frame
                new_vec = (np.sin(θo)*np.cos(φo)*x +
                        np.sin(θo)*np.sin(φo)*y +
                        np.cos(θo)*z)
                new_dirs[j] = new_vec / np.linalg.norm(new_vec)

            # commit back to rays array
            θ_new = np.degrees(np.arccos(np.clip(new_dirs[:,2], -1, 1)))
            φ_new = (np.degrees(np.arctan2(new_dirs[:,1], new_dirs[:,0])) % 360)
            rays['theta_deg'][idxs] = θ_new
            rays['phi_deg'][idxs]   = φ_new

        # Reset mfp_scattering for scattered rays
        rays['mfp_scattering'][to_scatter] = mfp_scattering(rays['wavelength_nm'][to_scatter])




        # === Count events ===
        total_absorptions += np.sum(to_absorb)
        total_scatters += np.sum(to_scatter)

        # Check if any rays are alive but outside z bounds
        in_bounds_mask = (
            (rays['z'] >= 0) & (rays['z'] <= 100000) &
            (rays['x'] >= 0) & (rays['x'] <= 10e30) &
            (rays['y'] >= 0) & (rays['y'] <= 10e30)
        )
        alive_and_in_bounds = rays['alive'] & in_bounds_mask

                # Diagnostic: check for any suspicious MFPs above 420 nm
        mask_overshoot = sampled_wavelengths > 420
        suspicious_mfp = mfp_absorption[mask_overshoot] < 1e6  # threshold: ~10 mm


        #bad_abs = to_absorb & (rays['wavelength_nm'] > 420)
        #if np.any(bad_abs):
        #    print("❌ Absorption triggered at forbidden λ > 420 nm!")
        #    for wl in rays['wavelength_nm'][bad_abs][:10]:
        #        print(f"  λ = {wl:.1f} nm")

        # Diagnostic: Average MFPs of currently alive rays
        alive_mask = rays['alive']

        if np.any(alive_mask):
            avg_mfp_abs = np.mean(rays['mfp_absorption'][alive_mask])
            avg_mfp_scat = np.mean(rays['mfp_scattering'][alive_mask])
            #print(f"Step {step:>4}: ⟨MFP_scat⟩ = {avg_mfp_scat:9.2f} nm")

        # Diagnostic: Z-distribution of alive rays
        alive_z = rays['z'][rays['alive']]

        if len(alive_z) > 0:
            hist, bin_edges = np.histogram(alive_z, bins=10, range=(0, 100000))
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            print(f"Step {step:>4}: Z-distribution of alive rays:")
            for z, count in zip(bin_centers, hist):
                print(f"   z = {z:6.0f} nm: {count:6d} rays")

        # Add total path traversed (step size * 1 for every step taken by each alive ray)
        rays['path_traversed'][still_inside] += step_size
        avg_path_um = np.mean(rays['path_traversed'][rays['alive']]) / 100000
        print(f"Step {step}: Avg path traversed by alive rays: {avg_path_um:.2f} × 100µm")

            # Diagnostic: Average MFPs of currently alive rays
        alive_mask = rays['alive']

        if np.any(alive_mask):
            avg_mfp_abs = np.mean(rays['mfp_absorption'][alive_mask])
            avg_mfp_scat = np.mean(rays['mfp_scattering'][alive_mask])
            print(f"Step {step}: ⟨MFP_abs⟩ = {avg_mfp_abs:.2e} nm, ⟨MFP_scat⟩ = {avg_mfp_scat:.2e} nm")

        print(f"Step {step}: Scattered this step = {n_scatter}, Total scattered so far = {total_scatters}")
                
        pbar.n = N_rays - np.sum(rays['alive'])
        pbar.refresh()

# --- Example output ---
print(rays[:3])

# Diagnostics for transmitted wavelengths
transmitted_wavelengths = rays['wavelength_nm'][rays['transmitted'] & ~rays['initial']]
print("Sample of transmitted wavelengths:", transmitted_wavelengths[:10])
print("Number of transmitted:", len(transmitted_wavelengths))
print("Min wavelength:", np.min(transmitted_wavelengths) if len(transmitted_wavelengths) > 0 else 'N/A')
print("Max wavelength:", np.max(transmitted_wavelengths) if len(transmitted_wavelengths) > 0 else 'N/A')

print(f"Final fraction of rays transmitted: {np.mean(rays['transmitted']):.4f}")


# Calculate common wavelength bins
min_wave = 300
max_wave = 1000
bins = np.linspace(min_wave - 5, max_wave + 5, 400)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Original AM1.5 solar spectrum (resample to same bin centers for fair plotting)
solar_hist, _ = np.histogram(sampled_wavelengths, bins=bins)

# All photons transmitted (incl. re-emitted)
all_transmitted_wavelengths = rays['wavelength_nm'][rays['transmitted']]

# Initial photons that were lost (absorbed and not re-emitted)
initial_lost_mask = rays['initial'] & ~rays['alive'] & ~rays['transmitted']
initial_lost_wavelengths = rays['initial_wavelength_nm'][initial_lost_mask]


# Histograms
hist_solar_input = solar_hist
hist_all_transmitted, _ = np.histogram(all_transmitted_wavelengths, bins=bins)
hist_lost, _ = np.histogram(initial_lost_wavelengths, bins=bins)

plt.figure(figsize=(12, 6))
# Restrict plot to 800 nm (already correct)
plt.plot(
    bin_centers[bin_centers <= 800],
    hist_solar_input[bin_centers <= 800] / 1e6,
    label=f'Original Solar Spectrum, θ={theta_deg:.0f}°',
    linestyle=':', color='black'
)
plt.plot(
    bin_centers[bin_centers <= 800],
    hist_all_transmitted[bin_centers <= 800] / 1e6,
    label='Transmitted',
    linewidth=1.5
)
plt.xlabel('Wavelength (nm)')
# Updated y-axis label (LaTeX-style)
plt.ylabel(r'$\mathrm{Photons} \,/\, (\mathrm{m}^2 \cdot \mathrm{s})$ (a.u.)')
# Dynamic title with scattering label
scatter_label = "with scattering" if ENABLE_SCATTERING else "without scattering"
plt.title(f'Spectrum for n = 1.5; flake thickness = 100 nm; QY = {quantum_yield}; {scatter_label}')
plt.legend()
plt.grid(True)
plt.tight_layout()
filename = f"spectrum_n{n1}_thick100_QY{quantum_yield}_{'scattering' if ENABLE_SCATTERING else 'no_scattering'}.png"
plt.savefig(filename)


# Rays that were absorbed (i.e., dead, not transmitted, and initial)
absorbed_mask = rays['initial'] & ~rays['alive'] & ~rays['transmitted']
absorbed_wavelengths = rays['initial_wavelength_nm'][absorbed_mask]
absorbed_wl = rays['initial_wavelength_nm'][rays['absorbed']]
print("Min absorbed λ:", np.min(absorbed_wl))
print("Max absorbed λ:", np.max(absorbed_wl))


# --- Histogram of Absorption Events by Wavelength ---

# Get wavelengths of absorbed (and not reemitted) rays
absorbed_mask = rays['initial'] & ~rays['alive'] & ~rays['transmitted']
absorbed_wavelengths = rays['initial_wavelength_nm'][absorbed_mask]

# --- Plot average MFP for each initial wavelength ---

# Only consider initial rays (not reemitted ones)
initial_rays = rays[rays['initial']]

# Group by initial wavelength (rounded to nearest nm)
wavelengths = initial_rays['initial_wavelength_nm'].astype(int)
mfp_values = initial_rays['mfp_absorption']

# Compute average MFP per wavelength
unique_wl = np.unique(wavelengths)
avg_mfp = [np.mean(mfp_values[wavelengths == wl]) for wl in unique_wl]

# Use initial sampled wavelengths and their computed MFPs directly
initial_wavelengths = sampled_wavelengths  # before assignment to rays
initial_mfp_values = mfp_absorption        # as computed right after sampling

# Group and average
unique_wl = np.unique(initial_wavelengths)
avg_mfp = [np.mean(initial_mfp_values[initial_wavelengths == wl]) for wl in unique_wl]



# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(absorbed_wavelengths, bins=100, color='crimson', edgecolor='black')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorption Count")
plt.title("Number of Absorptions vs. Wavelength")
plt.grid(True)
plt.tight_layout()
plt.savefig("absorption_histogram.png")

# at very end, after the while‐loop:
finished   = ~rays['alive']            # rays that have died or exited
final_reflect  = finished & rays['reflected'] & ~rays['transmitted'] & ~rays['absorbed']
final_transmit = finished & rays['transmitted']
final_absorb   = finished & rays['absorbed']
still_alive    = rays['alive']

n_refl  = final_reflect.sum()
n_trans = final_transmit.sum()
n_abs   = final_absorb.sum()
n_alive = still_alive.sum()

print("FINAL COUNTS:")
print(f"  Reflected   = {n_refl:6d} / {N_rays}  ({n_refl/N_rays:.2%})")
print(f"  Transmitted = {n_trans:6d} / {N_rays}  ({n_trans/N_rays:.2%})")
print(f"  Absorbed    = {n_abs:6d} / {N_rays}  ({n_abs/N_rays:.2%})")
print(f"  Still alive = {n_alive:6d} / {N_rays}  ({n_alive/N_rays:.2%})")
print("  → sum        =", n_refl + n_trans + n_abs + n_alive)

# after your existing histograms...
# bin_centers, hist_solar_input, hist_all_transmitted already defined

transmittance = hist_all_transmitted / np.clip(hist_solar_input, 1, np.inf)


direct_tx = np.sum(rays['transmitted'] & rays['initial'])
reemitted_tx = np.sum(rays['transmitted'] & ~rays['initial'])
print(f"Direct transmitted: {direct_tx} (≈{direct_tx/N_rays:.1%})")
print(f"Re-emitted bottom-exit: {reemitted_tx} (≈{reemitted_tx/N_rays:.1%})")



# === η_perovskite: total useful photon efficiency (includes losses) ===
# Interpolate McCree quantum yield function (normalized)
mccree_interp = interp1d(mccree_df["Wavelength_nm"], mccree_df["Relative_Quantum_Yield_CO2"],
                         bounds_error=False, fill_value=0)

# Use 300–800 nm range to match the simulation
bins_eta = np.arange(300, 801)
bin_centers_eta = (bins_eta[:-1] + bins_eta[1:]) / 2

# Input spectrum histogram
hist_input_eta, _ = np.histogram(sampled_wavelengths, bins=bins_eta)

# Transmitted spectrum histogram (all transmitted, direct + reemitted if any)
transmitted_wavelengths_all = rays["wavelength_nm"][rays["transmitted"]]
hist_trans_eta, _ = np.histogram(transmitted_wavelengths_all, bins=bins_eta)

# McCree weights at bin centers
w = mccree_interp(bin_centers_eta)

# Total useful photons in and out
U_in = np.sum(hist_input_eta * w)
U_out = np.sum(hist_trans_eta * w)

# Define eta_perovskite as the overall useful-photon ratio (can be < 1 when there are losses)
eta_perovskite = U_out / np.clip(U_in, 1e-12, np.inf)

print(f"η_perovskite (useful photon ratio) = {eta_perovskite:.4f}")

