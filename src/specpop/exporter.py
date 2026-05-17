import pandas as pd
import numpy as np
import os

class Exporter:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_results(self, results, result_filename='intensities.csv'):
        df_results = pd.DataFrame(results)
        
        # Ensure exact column structure for Intensity Output
        cols = [
            'n_upper', 'nist_wavelength', 'observed_wavelength', 
            'integrated_intensity', 'A_ki', 'g_sub', 'g_lumped', 
            'density_sublevel', 'averaged_densities', 'relative_error'
        ]
        
        # Map existing column names to requested ones if necessary
        column_mapping = {
            'Observed_Wavelength': 'observed_wavelength',
            'Integrated_Intensity': 'integrated_intensity'
        }
        df_results = df_results.rename(columns=column_mapping)
        
        # Add missing columns with NaN
        for col in cols:
            if col not in df_results.columns:
                df_results[col] = np.nan
        
        # --- Apply Formatting ---
        # Round Wavelengths to two decimals
        for col in ['nist_wavelength', 'observed_wavelength']:
            if col in df_results.columns:
                df_results[col] = df_results[col].round(2)
        
        # Integers: n_upper, g_sub, g_lumped, integrated_intensity, A_ki
        int_cols = ['n_upper', 'g_sub', 'g_lumped', 'integrated_intensity', 'A_ki']
        for col in int_cols:
            if col in df_results.columns:
                # Use Int64 to handle NaNs while keeping integers
                df_results[col] = pd.to_numeric(df_results[col], errors='coerce').round(0).astype('Int64')
        
        # Round densities to 4 significant digits
        def round_sig(x, sig=4):
            if pd.isna(x) or x == 0:
                return x
            return float(f'{x:.{sig}g}')

        for col in ['density_sublevel', 'averaged_densities']:
            if col in df_results.columns:
                df_results[col] = df_results[col].apply(round_sig)
        
        # Relative error to two decimals
        if 'relative_error' in df_results.columns:
            df_results['relative_error'] = df_results['relative_error'].round(2)
                
        df_results = df_results[cols]
        
        # Sort results: n_upper first, averaged_densities second, nist_wavelength third
        df_results = df_results.sort_values(by=['n_upper', 'averaged_densities', 'nist_wavelength'], 
                                            ascending=[True, True, True])
        
        csv_path = os.path.join(self.output_dir, result_filename)
        df_results.to_csv(csv_path, index=False)
        print(f"\nAll lines analyzed! Summary saved to {csv_path}")

    def save_aggregated_results(self, aggregated_results, filename='aggregated_densities.csv'):
        df_agg = pd.DataFrame(aggregated_results)
        
        # Ensure exact column structure for Aggregated Output
        cols = ['n_upper', 'g_lines', 'g_lumped', 'lumped_relative_density']
        
        # Add missing columns with NaN
        for col in cols:
            if col not in df_agg.columns:
                df_agg[col] = np.nan

        # --- Apply Formatting ---
        # 1. Integers: n_upper, g_lumped
        int_cols = ['n_upper', 'g_lumped']
        for col in int_cols:
            if col in df_agg.columns:
                df_agg[col] = pd.to_numeric(df_agg[col], errors='coerce').round(0).astype('Int64')

        # 2. Round lumped_relative_density to 4 significant digits
        def round_sig(x, sig=4):
            if pd.isna(x) or x == 0:
                return x
            return float(f'{x:.{sig}g}')

        if 'lumped_relative_density' in df_agg.columns:
            df_agg['lumped_relative_density'] = df_agg['lumped_relative_density'].apply(round_sig)
        
        df_agg = df_agg[cols]
        
        csv_path = os.path.join(self.output_dir, filename)
        df_agg.to_csv(csv_path, index=False)
        print(f"Aggregated number densities saved to {csv_path}")
