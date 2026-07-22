# Harvest Timing Demo

## Objective
Use PINN to predict optimal harvest time for Ac-225 in a recycler scenario.

## Key Results
- **Optimal harvest time**: ~416 hours (17.3 days)
- **Peak Ac-225 yield**: 3.03e+18 atoms
- **Flux sensitivity**: no consistent flux→peak-time trend resolved in this window

## Application
- Operators can use PINN to predict harvest windows **without solving ODEs** at query time
- Enables real-time optimization of irradiation schedules
- Fast inference (milliseconds) vs. minutes for numerical ODE solve

