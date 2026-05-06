import argparse
import os
from .interface import Interface

def main():
    """
    Main entry point for the spectral analysis package.
    
    Parses command-line arguments to get the path to the configuration file
    and runs the analysis.
    """
    parser = argparse.ArgumentParser(
        description="Perform Optical Emission Spectroscopy (OES) analysis based on a configuration file."
    )
    parser.add_argument("--config",help="Path to the TOML configuration file for the analysis.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at '{args.config}'")
        return

    # Create and run the analysis interface
    interface = Interface(args.config)
    interface.run_analysis()

if __name__ == '__main__':
    main()
