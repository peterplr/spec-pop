# specpop - Optical Emission Spectroscopy (OES) Analysis Tool

This is a Python-based tool for analyzing Optical Emission Spectroscopy (OES) data. It automates the process of identifying spectral lines, calculating their integrated intensities (with automatic baseline and dark spectrum subtraction), and cross-referencing them against the NIST atomic spectra database. Furthermore, it groups identified lines and calculates lumped relative sublevel densities based on Vlcek's collisional-radiative model for Argon.

## Features
- Auto-detection of spectral lines above a defined threshold.
- Manual mode for integrating user-defined peak lists.
- Automatic full-width half-maximum (FWHM) calculation for defining integration windows.
- Cross-referencing detected lines with NIST database to obtain transition probabilities (A_ki) and statistical weights.
- Ambiguity detection: Prevents misidentification when multiple valid NIST lines fall within the instrument's optical resolution.
- Extrapolation of sublevel densities to lumped level densities based on Vlcek's CRM.

## Installation

It is recommended to use a virtual environment to install dependencies.

```bash
# Create and activate a virtual environment
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install the package in editable mode
pip install -e .
```

## Configuration
The analysis behavior is entirely controlled via a `config.toml` file. This includes defining paths to your raw spectra and dark current files, as well as tuning parameters for peak detection, baseline integration, and graph output.

Check the provided `configs/config.toml` file for a fully documented example.

## Usage

Once installed, the package provides a command-line executable.

Run the analysis by pointing the tool to your configuration file:

```bash
specpop --config path/to/your/config.toml
```

## Outputs
The tool will generate:
- `intensities.csv`: A detailed breakdown of every matched spectral line, its integrated intensity, matching NIST wavelength, and specific transition parameters.
- `aggregated_densities.csv`: A high-level summary of relative densities grouped by Vlcek's lumped levels, including calculated relative errors.
- Graphical plots: Individual plots showing the baseline and integration area for every analyzed line, and an overarching plot displaying the full spectrum with marked integration zones.