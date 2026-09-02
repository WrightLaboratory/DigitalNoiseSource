# Digital Noise Source (DNS) Analysis Framework

Analysis framework for precision timing and phase characterization of drone-based calibration measurements. This repository contains code and notebooks for processing drone flight data alongside correlator measurements to validate timing precision and phase stability.

## Project Structure


## Overview

The analysis pipeline:
1. Loads drone flight telemetry and correlator data
2. Synchronizes the two datasets via timestamp alignment
3. Extracts phase and amplitude measurements
4. Analyzes timing precision and stability across different timescales

Key notebooks demonstrate precision timing characterization using Furuno 8804 clocks and phase analysis from drone-based measurements.

## Core Modules

**concat.py** – Data alignment and stitching
- Loads drone CSV files and correlator HDF5 data
- Synchronizes timestamps between datasets
- Interpolates drone positions to correlator time grid

**drone.py** – Geometry calculations
- Computes drone angle and distance relative to ground equipment
- Converts GPS coordinates to local Cartesian frame

**time_utils.py** – Clock and timing analysis
- Measures clock drift and jitter
- Corrects timing discontinuities
- Computes drift/jitter metrics per integration window

**plotting_utils.py, fitting_utils.py** – Visualization and statistics

## Getting Started

### Prerequisites
- Python 3.8+
- NumPy, SciPy, Pandas, Matplotlib, Scikit-learn
- h5py (for correlator data), PyGeodesy (for geometry)
- Jupyter (for running notebooks)

### Installation
```bash
git clone https://github.com/WrightLaboratory/DigitalNoiseSource.git
cd DigitalNoiseSource
pip install numpy scipy pandas matplotlib scikit-learn astropy h5py pygeodesy
```

### Running the Analysis

1. Update metadata YAML files with your site geometry and flight parameters
2. Ensure correlator and drone data files are accessible
3. Run notebooks in `notebooks/phase_analysis/` in order:
   - Clock characterization (Furuno notebook)
   - Phase analysis (slowpass, then timescale variants)
   - Timing calibration (interpolation notebook)

Each notebook generates plots and CSV/TXT output files documenting the results.

## Key Results

- Validates timing precision to sub-nanosecond levels
- Demonstrates phase stability across multiple integration timescales
- Documents drone-based calibration performance

## Contact

Wright Laboratory, Yale University
