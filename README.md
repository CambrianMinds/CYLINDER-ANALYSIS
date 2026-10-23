# Martian Cylinder Forensic Analysis | NASA Curiosity Sol 3556

### Sub-Pixel Photogrammetry, CAHVOR Optical Ray-Tracing, and 3D Morphometry of Curiosity's Most Enigmatic Surface Feature

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PDS4 Validated](https://img.shields.io/badge/NASA_PDS-EDR_Lossless-orange.svg)](https://pds-imaging.jpl.nasa.gov/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Manuscript](https://img.shields.io/badge/Manuscript-PDF_(13_Pages)-red.svg)](martian_cylinder_analysis.pdf)

---

![Figure 2: Sobel Edge-Gradient & Flank Tangents](figures/sobel_gradient_and_flank_tangents.png)

---

## Executive Summary

On August 7, 2022 (**Sol 3556** of its mission), NASA’s Mars Science Laboratory (*Curiosity*) captured frame `3556MR1025170721700585C00_DXXX` using its Mast Camera Right-eye (Mastcam-100, $f = 100\,\text{mm}$) during a routine reconnaissance mosaic in **Paraitepuy Pass**, Mount Sharp, Gale Crater.

Resting amid the crumbly sulfate-bearing shale was an anomalous, highly regular cylindrical feature with an apparently planar circular end-cap and razor-sharp parallel flanks. For years, debate has raged online between claims of "extraterrestrial artifacts" and dismissals of "compression artifacts and pareidolia."

This repository contains the **first complete, sub-pixel mathematical re-analysis** derived directly from the uncompressed, lossless **12-bit NASA Planetary Data System (PDS) Experiment Data Record (EDR)**. Using the camera's calibrated **CAHVOR optical model**, orbital ephemerides, solar vector physics, and unconstrained conic geometry, we demonstrate that the object is an empirical physical reality whose geometric properties satisfy Euclidean cylinder equations with sub-millimeter precision.

---

## Forensic Measurement Summary

| Parameter | Measured Value | Uncertainty ($\pm 1\sigma$) | Physical / Geometric Interpretation |
| :--- | :--- | :--- | :--- |
| **PDS Product ID** | `3556MR1025170721700585C00_DXXX` | — | Mastcam-100 Lossless Raw EDR |
| **Camera Elevation Tilt** | $-21.84^\circ$ | $\pm 0.05^\circ$ | Mast pointing relative to rover chassis |
| **Rover Pitch / Roll Tilt** | $14.84^\circ$ | $\pm 0.10^\circ$ | Chassis inclination on sulfate slope |
| **Line-of-Sight Depression** | $23.99^\circ$ | $\pm 0.15^\circ$ | Camera-to-target angle relative to local gravity |
| **Slant Range (Standoff $d$)** | **$5.168\,\text{m}$** | $\pm 0.12\,\text{m}$ ($\pm 2.3\%$) | Exact CAHVOR ray-intersection distance |
| **Calibrated Sensor Scale** | **$0.3865\,\text{mm / pixel}$** | $\pm 0.009\,\text{mm/px}$ | Optical resolution at target plane |
| **Front Outer Rim Diameter ($D$)** | **$15.5 \pm 0.4\,\text{mm}$** | $\pm 0.4\,\text{mm}$ ($1.55\,\text{cm}$) | Exactly $0.61\,\text{inches}$ (Standard aerospace casing) |
| **Front De-projected Radius ($R_1$)** | $26.91\,\text{px}$ (crop space) | $\pm 0.28\,\text{px}$ | Residual from perfect circle: $\mathbf{\sigma_r = 0.74\,\text{px}}$ ($\pm 1.4\%$) |
| **Rear Terminus Radius ($R_2$)** | $10.20\,\text{px}$ (crop space) | $\pm 0.35\,\text{px}$ | Far end-cap curvature ($37.9\%$ apparent scale) |
| **Perspective Depth Recession** | $Z_{\text{rear}} / Z_{\text{front}} = 2.64\times$ | $\pm 0.08$ | Strict pinhole perspective scaling ($R \propto 1/Z$) |
| **Projected Visible Length** | $22.0 \pm 0.5\,\text{mm}$ | $\pm 0.5\,\text{mm}$ ($2.20\,\text{cm}$) | $77.4\,\text{px}$ across detector surface |
| **True 3D Physical Length** | **$47.1 \pm 1.1\,\text{mm}$** | $\pm 1.1\,\text{mm}$ ($4.71\,\text{cm}$) | De-projected true length along longitudinal core |
| **True Aspect Ratio ($L/D$)** | **$3.04 \pm 0.08$** | $\pm 0.08$ | Classic elongated right circular cylinder |
| **Regolith Contact Line** | $y = -0.410\,x + 201.50$ | $-22.3^\circ$ slope | Physical soil horizon crossing cylinder |
| **Exposed Face Fraction** | **$68.8\%$** | $\pm 1.5\%$ | Subaerial sunlit / self-shadowed sector |
| **Buried Subsurface Fraction** | **$31.2\%$** | $\pm 1.5\%$ | **$7.3\,\text{mm}$ embedded** into compacted regolith |

---

## 3D Hull Reconstruction & Physical Optics

![Figure 5: 3D Hull Reconstruction](figures/cylinder_3d_hull_reconstruction.png)

1. **Optical Self-Shadow vs. Ground Shadow:** At Sol 3556 solar azimuth ($140.4^\circ$) and elevation ($50.4^\circ$), the planar circular face points away from the solar vector, placing it in deep **optical self-shadow** ($63\,\text{DN}$). Solar grazing over the top and right hull casts a sharp **ground shadow** ($66\,\text{DN}$) extending into the regolith.
2. **De-projecting the Circle:** Fitting an unconstrained conic to the front rim yields an ellipse with major/minor axes $a = 25.96\,\text{px}$, $b = 16.45\,\text{px}$, rotated $-54^\circ$. Inverting the viewing projection yields a **mathematically circular disc with a mean radial error of just $0.74\,\text{pixels}$ ($\pm 1.4\%$)**, matching the optical point-spread function of the M-100 optics.
3. **Flank Tangency & Zero Free Parameters:** The upper longitudinal boundary connects the apex of the front face to the apex of the rear terminus with sub-pixel collinearity. Four independent constraints (front circle + rear circle + flank tangency + soil horizon) converge on a single closed 3D cylinder model with **zero free parameters**.

---

## Repository Structure

```text
├── data/                                         # Official PDS4 EDR Raw Telemetry
│   ├── 3556MR1025170721700585C00_DXXX.IMG        # 12-bit binary camera EDR array
│   ├── 3556mr1025170721700585c00_dxxx.xml        # Official PDS4 XML label & CAHVOR model
│   ├── profiles.json                             # 1D sub-pixel intensity profiles
│   └── analysis_results.json                     # Structured JSON metrics & edge coordinates
│
├── figures/                                      # Full Resolution Publication Figures (15 assets)
│   ├── pds_raw_full.png                          # Unprocessed 1200x1200 Mastcam EDR frame
│   ├── pds_full_with_callout.png                 # Figure 1(a): Context frame with callout box
│   ├── cylinder_crop_raw.png                     # Figure 1(b): Native sensor crop
│   ├── cylinder_red_shadow_blue_clean.png        # Figure 1(c): Mathematical delineation
│   ├── sobel_gradient_and_flank_tangents.png     # Figure 2: Gradient ridges & tangents
│   ├── pds_edr_native_forensics.png              # Figure 3: EDR integer level forensics
│   ├── straight_on_deprojection_proof.png        # Figure 4: Deprojection & aspect ratio
│   ├── cylinder_3d_hull_reconstruction.png       # Figure 5: 3D perspective hull reconstruction
│   └── scientific_dashboard_red_blue.png         # Figure 6: Master 6-panel diagnostic matrix
│
├── scripts/                                      # Complete Python Forensic Pipeline (14 Scripts)
│   ├── inspect_telemetry.py                      # PDS4 XML label parser & CAHVOR extractor
│   ├── check_site_frame.py                       # Mars rover-to-site coordinate transforms
│   ├── run_telemetry_forensics.py                # CAHVOR ray-tracing, standoff & error bounds
│   ├── solve_exact_standoff.py                   # Ground distance & sensor depression geometry
│   ├── solve_shadow_and_slope.py                 # Solar azimuth & shadow geometry verification
│   ├── cv_cylinder_analysis.py                   # Sub-pixel Sobel gradient & profile extraction
│   ├── reconstruct_3d_hull.py                    # 3D cylinder hull modeling & perspective solver
│   ├── generate_figure1_context.py               # Generates Figure 1(a) context frame
│   ├── generate_clean_delineation.py             # Generates Figure 1(c) clean delineation
│   ├── generate_clean_figure2.py                 # Generates Figure 2 Sobel & flank tangents
│   ├── generate_front_on_view.py                 # Generates front-on projective transformation
│   ├── generate_clean_dashboard.py               # Generates Figure 6 scientific dashboard
│   ├── generate_physical_decomposition_figure.py # Generates physical decomposition proof
│   └── generate_camera_orbit.py                  # Generates 3D orbit visualization
│
├── martian_cylinder_analysis.tex                 # Master publication LaTeX source (13 pages)
├── martian_cylinder_analysis.pdf                 # Compiled publication PDF (0 errors, 0 warnings)
├── martian_cylinder_article.md                   # Long-form Markdown investigative article
├── requirements.txt                              # Python environment dependencies
└── README.md                                     # Project documentation
```

---

## Quickstart & Replication

### 1. Installation

Clone this repository and install the verified dependencies:

```bash
git clone https://github.com/<your-username>/martian-cylinder-sol3556.git
cd martian-cylinder-sol3556

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Extract Official PDS4 Telemetry & Photogrammetry

```bash
# Parse PDS4 XML, Rover Site Frame, and CAHVOR camera vectors
python scripts/inspect_telemetry.py
python scripts/check_site_frame.py

# Calculate rigorous standoff distance (5.168 m) and margin of error (+-2.3%)
python scripts/run_telemetry_forensics.py
```

### 3. Run Computer Vision & Gradient Extraction

```bash
# Sub-pixel Sobel edge-gradient detection and bootstrap resampling
python scripts/cv_cylinder_analysis.py

# Solve 3D perspective hull reconstruction and regolith intersection
python scripts/reconstruct_3d_hull.py
```

### 4. Regenerate All Publication Figures

```bash
python scripts/generate_figure1_context.py
python scripts/generate_clean_delineation.py
python scripts/generate_clean_figure2.py
python scripts/generate_front_on_view.py
python scripts/generate_clean_dashboard.py
```

### 5. Compile the LaTeX Manuscript

```bash
pdflatex -interaction=nonstopmode martian_cylinder_analysis.tex
```

---

## Hypotheses & Forensic Discussion

1. **Natural Geological Ventifact:**  
   Wind-driven sand in Gale Crater routinely produces faceted ventifacts. However, aeolian ventifaction follows aerodynamic streamlines, creating teardrop flutes and asymmetric keels. Concretion nodules are spherical, and columnar basalt prisms are hexagonal/pentagonal. A natural process grinding an isolated, hollow-compatible right circular cylinder with an orthogonal circular face ($\sigma_r = 0.74\,\text{px}$) on a bed of crumbling sulfate shale represents an extraordinary statistical outlier.

2. **Discarded Spacecraft Hardware:**  
   At $15.5\,\text{mm}$ outer diameter and $47.1\,\text{mm}$ length ($0.61 \times 1.86\,\text{inches}$), the object precisely matches the standard form factor of aerospace mechanism components:
   * A **pyro-actuator cartridge casing or pin-puller shroud** from stage separation.
   * A **drill bit sleeve or calibration collar** from Curiosity's robotic arm (SA/SPaH).
   * A **machined structural bushing or standoff pin**.  
   *Dilemma:* Bradbury Landing (touchdown) is $20\,\text{km}$ north. If shed from the rover chassis during traverse, NASA engineering anomaly logs record no lost hardware sleeves, and the object exhibits no sheared mounting ears or electrical leads. Furthermore, **$31.2\%$ of its front face is embedded $7.3\,\text{mm}$ deep** into compacted regolith, demonstrating surface stability over time.

3. **Artifact of Unknown Provenance:**  
   If known geological mechanisms cannot account for the planar circular geometry and the object cannot be reconciled with documented rover jettison events, empirical discipline requires keeping the anomalous hypothesis open. Curiosity carries contact instruments capable of resolving this question definitively: **MAHLI** (micron-scale optical imaging of metallic tool marks/crystallinity), **APXS** (elemental alloy composition), and **ChemCam** (remote LIBS laser spectroscopy).

---

## Citation & Academic Reference

If you cite or utilize this dataset, telemetry pipeline, or photogrammetric models in your research:

```bibtex
@article{bogner2026martiancylinder,
  title={Sub-Pixel Morphometry, Photogrammetric Reconstruction, and Multi-Hypothesis Forensic Evaluation of an Anomalous Cylindrical Feature in Curiosity Sol 3556 Mastcam Telemetry},
  author={Bogner, Justin},
  journal={Planetary Surface Analysis Working Paper Series},
  year={2026},
  url={https://github.com/cambrianminds/cylinder-analysis}
}
```

---

## Dedication

Dedicated to **Richard Sam Bogner** (1931–2015), Battery Systems Engineer at NASA’s Jet Propulsion Laboratory (JPL). His pioneering work on the **Apollo** program and the early **Mariner** missions helped lay the foundations for humanity’s exploration of Mars.

*In space exploration, opinions are cheap—the telemetry never lies.*
