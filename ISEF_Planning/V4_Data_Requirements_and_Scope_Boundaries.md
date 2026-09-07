# V4 Data Requirements, Experimental Validation & Claim Boundaries

## 1. Highest Priority: Berkeley Experimental Validation Checklist

To compare transport simulations and surrogate models directly against the 33 and 40 MeV Berkeley cyclotron experiments, the following finite items are tracked:

1. **Target-Integrated Neutron Fluence/Spectrum:** Energy-binned fluence entering the target capsule, including units and source normalization.
2. **Target Geometry & Encapsulation:** Ra-226 sample dimensions (1 mg radium nitrate), capsule material composition, wall thickness, and distance from the Beryllium converter.
3. **Beam Characteristics & History:** Deuteron beam energy (33/40 MeV), beamspot diameter, target offset, and time-dependent beam current or total integrated charge.
4. **Experimental Timestamps:** Precise timestamps for start/stop of irradiation, end of beam (EOB), radiochemical dissolution, chemical separation, and gamma spectroscopy counting.
5. **Measured Isotope Activities:** Measured Actinium-225 activities, counting uncertainties (1-sigma), detector efficiencies, and the exact decay-correction reference timestamp.
6. **Radiochemical Separation Efficiency:** Chemical recovery yield percentage and associated experimental uncertainty.
7. **Numerical Actinium-227 Detection Limit:** Quantitative minimum detectable activity (MDA) or upper-bound threshold, rather than qualitative "none detected" statements.
8. **Uncertainty & Covariance Data:** Energy-dependent cross-section uncertainties, target mass uncertainties, and detector calibration covariances.

---

## 2. Requirements for OpenMC Computational Teacher Dataset

When generating synthetic transport training data, the following standards must be preserved:

* **Material & Geometry Definition:** Explicit material cards with isotopic densities and verified 3D bounding surfaces.
* **Nuclear Data Libraries:** Evaluated high-energy libraries (e.g., ENDF/B-VIII.0, TENDL, or JENDL-5) extending across the full 0 to 40 MeV neutron energy domain.
* **Tally Specifications:** Energy-integrated and spatially resolved `(n,2n)`, `(n,gamma)`, and impurity production reaction tallies with relative Monte Carlo standard deviations (< 1%).
* **Provenance & Reproducibility:** Fixed random seeds, particle histories (>= 10^6 per batch), runtime logs, and exact input configuration hashes.
* **Geometry Splitting:** Explicitly partitioned train, calibration, and untouched out-of-distribution test geometry families.

---

## 3. Strict Process Scope & Claim Boundaries

To ensure complete scientific integrity, claims must adhere to the following boundaries:

* **Supported Claim:** "Accelerated computational schedule screening and surrogate approximation of transport-coupled reaction rates."
* **Unsupported Claim (Do Not Make):** "Making Actinium-225 production cheaper or faster end-to-end commercially."
  * *Reason:* A whole-process claim requires measuring radiochemical dissolution time, column purification recovery losses, regulatory quality-control release testing, target recovery and recycling, hot-cell labor, waste management, and cold-chain logistics costs.

---

## 4. The 4 Focused Expert Data Requests

When contacting researchers for Berkeley experimental data, focus strictly on these four specific requests:

1. Target-integrated neutron spectrum/fluence with units and normalization.
2. Target geometry, capsule materials, and beam current/charge history.
3. Measurement timing, chemical recovery efficiency, and activity uncertainties.
4. Numerical Actinium-227 minimum detectable activity (MDA) threshold.

*Rule:* If any parameters remain proprietary or unavailable, document the request and record the missing parameter as an explicit scientific limitation in the manuscript.
