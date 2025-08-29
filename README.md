# spectral-conversion-foil-light-transport
Monte Carlo ray-tracing framework for modeling light transport, scattering, and re-emission in perovskite flakes embedded in a polymer matrix.

# Solar Foil Monte Carlo Simulation

This repository contains a Monte Carlo ray-tracing framework for modeling light transport, scattering, and re-emission in perovskite flakes embedded in a polymer matrix, for spectral tuning in agricultural applications.  

The code simulates:
- Absorption of UV/near-UV photons
- Scattering using Mie/T-matrix theory
- Re-emission into the photosynthetically active radiation (PAR) range
- Transmission efficiency for different foil parameters

## Features
- Full photon path tracking
- Customizable perovskite concentration, flake thickness, refractive index, and quantum yield
- Modular design for extension (e.g., dome structures, new scatterers)

## Overview

The code uses:
- **Absorption spectrum file** → Defines how UV/visible light is absorbed by the perovskite foil.  
- **Emission spectrum file** → Defines how absorbed photons are re-emitted in the PAR (photosynthetically active radiation) range.  
- **Solar spectrum (AM1.5)** → Used as the incident light source.  
- **Scattering precomputation file** → Lookup table of scattering angles, generated before running the main simulation.  

The **main file (`main_code.py`)** ties everything together.  
It uses the solar spectrum, one absorption spectrum, one emission spectrum, and the precomputed scattering table to simulate light propagation.

---

## Workflow

1. **Precompute scattering angles**  
   Run the scattering precomputation script:  
   ```bash
   python Store_info_scatter.py

2. **Run the main simulation**  
   After generating the scattering file, run:  
   ```bash
   python main_code.py
