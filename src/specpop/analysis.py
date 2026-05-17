import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import warnings


class OESAnalysis:
    def __init__(self, data_path, line_data_provider=None, dark_image_path=None,
                 wl_col_name='Wavelength', count_col_name='Counts', delimiter='\t',
                 fwhm_multiplier=3.0, fwhm_search_window=10.0, default_width=1.0,
                 ignore_wavelengths=None):

        self.fwhm_multiplier = fwhm_multiplier
        self.fwhm_search_window = fwhm_search_window
        self.default_width = default_width
        self.ignore_wavelengths = ignore_wavelengths if ignore_wavelengths is not None else []

        # Inject the LineData class
        self.line_data = line_data_provider

        # Load Experimental Data
        self.data = pd.read_csv(data_path, sep=delimiter)
        self.wavelengths = self.data[wl_col_name].values
        self.counts = self.data[count_col_name].values.copy()

        if dark_image_path and os.path.exists(dark_image_path):
            dark_data = pd.read_csv(dark_image_path, sep=delimiter)
            self.counts -= dark_data[count_col_name].values
            self.counts = np.clip(self.counts, 0, None)

    def find_peak_properties(self, center_wl):
        """Finds properties of a peak near a given center wavelength."""
        # Use the larger fwhm_search_window to calculate peak width
        window_half_width_nm = self.fwhm_search_window / 2.0
        search_left = center_wl - window_half_width_nm
        search_right = center_wl + window_half_width_nm

        left_idx = (np.abs(self.wavelengths - search_left)).argmin()
        right_idx = (np.abs(self.wavelengths - search_right)).argmin()

        wl_window = self.wavelengths[left_idx:right_idx + 1]
        counts_window = self.counts[left_idx:right_idx + 1]

        if wl_window.size < 3:
            return center_wl, self.default_width, 0

        peaks_in_window, properties = find_peaks(counts_window, height=0, width=1)

        if peaks_in_window.size == 0:
            return center_wl, self.default_width, 0

        center_idx_in_window = (np.abs(wl_window - center_wl)).argmin()
        closest_peak_idx = (np.abs(peaks_in_window - center_idx_in_window)).argmin()

        actual_peak_idx = peaks_in_window[closest_peak_idx]
        width_in_samples = properties['widths'][closest_peak_idx]
        peak_height = properties['peak_heights'][closest_peak_idx]

        actual_peak_wl = wl_window[actual_peak_idx]
        avg_step = np.mean(np.diff(wl_window))
        width_in_nm = width_in_samples * avg_step

        return actual_peak_wl, width_in_nm, peak_height

    def calculate_line_intensity(self, center_wl, manual_width=None):
        """Calculates the true intensity of a single spectral line."""
        actual_peak_wl, fwhm_width, peak_height = self.find_peak_properties(center_wl)

        integration_window_width = manual_width if manual_width is not None else (self.fwhm_multiplier * fwhm_width)

        left_bound = actual_peak_wl - (integration_window_width / 2.0)
        right_bound = actual_peak_wl + (integration_window_width / 2.0)

        left_idx = (np.abs(self.wavelengths - left_bound)).argmin()
        right_idx = (np.abs(self.wavelengths - right_bound)).argmin()

        if left_idx >= right_idx:
            return 0, np.array([]), np.array([]), np.array([]), integration_window_width, actual_peak_wl, peak_height

        wl_window = self.wavelengths[left_idx:right_idx + 1]
        counts_window = self.counts[left_idx:right_idx + 1]

        wl_left, count_left = self.wavelengths[left_idx], self.counts[left_idx]
        wl_right, count_right = self.wavelengths[right_idx], self.counts[right_idx]

        slope = (count_right - count_left) / (wl_right - wl_left)
        baseline = count_left + slope * (wl_window - wl_left)

        corrected_counts = np.clip(counts_window - baseline, 0, None)
        integrated_intensity = np.trapezoid(y=corrected_counts, x=wl_window)

        return integrated_intensity, wl_window, counts_window, baseline, integration_window_width, actual_peak_wl, peak_height

    def process_target_list(self, target_peaks):
        """
        Feature 1: Process a specific list of peaks.
        target_peaks: List of dicts, e.g., [{'target_wl': 750.4, 'manual_width': 1.5}, ...]
        """
        results = []
        for peak in target_peaks:
            target_wl = peak.get('center')
            manual_width = peak.get('width', None)
            
            if target_wl is None:
                continue

            # Check if this wavelength is in the ignore list
            should_ignore = any(abs(target_wl - ignore_wl) < 0.1 for ignore_wl in self.ignore_wavelengths)
            if should_ignore:
                print(f"Info: Skipping manually targeted line {target_wl:.2f} nm as it is in the ignore_wavelengths list.")
                continue
            
            # Query the injected LineData class first to check for NIST ambiguity
            params = self.line_data.get_line_parameters(target_wl)
            
            # Print the info message if it exists (e.g. for a dominant line overriding ambiguity)
            if params.get('info_message'):
                print(params['info_message'])

            if params.get('is_ambiguous', False):
                print(f"Warning: Multiple NIST lines found within ambiguity window around target {target_wl:.2f} nm. Disregarding.")
                continue

            intensity, wl_win, counts_win, baseline, width, actual_wl, height = self.calculate_line_intensity(target_wl, manual_width)
            results.append({
                'Target_Wavelength': target_wl,
                'Observed_Wavelength': actual_wl,
                'Integrated_Intensity': intensity,
                'Peak_Height': height,
                'Match_Status': 'Matched', # Assume matched for manual processing for density calc
                'A_ki': params['A_ki'],
                'nist_wavelength': params['nist_wl'],
                'Upper_Level_Energy_Ek': params['E_k'],
                'n_upper': params['n_upper'],
                'g_sub': params['g_sub'],
                'g_lumped': params['g_lumped'],
                'is_metastable': params['is_metastable'],
                'upper_state_id': params['upper_state_id']
            })
        return pd.DataFrame(results)

    def auto_find_and_match(self, min_height, range_min=None, range_max=None, tolerance=1.0):
        if self.line_data is None:
            raise ValueError("A LineData instance must be provided to use auto-matching.")

        peaks, properties = find_peaks(self.counts, height=min_height, width=1)

        detected_peaks = []
        for i, peak_idx in enumerate(peaks):
            center_wl = self.wavelengths[peak_idx]

            if (range_min and center_wl < range_min) or (range_max and center_wl > range_max):
                continue
                
            # Check if this wavelength is in the ignore list (with a small 0.1nm tolerance for floating point matching)
            should_ignore = any(abs(center_wl - ignore_wl) < 0.1 for ignore_wl in self.ignore_wavelengths)
            if should_ignore:
                print(f"Info: Skipping detected peak near {center_wl:.2f} nm as it matches the ignore_wavelengths list.")
                continue
                
            intensity, _, _, _, _, actual_wl, peak_height = self.calculate_line_intensity(center_wl)

            # Query the injected LineData class
            params = self.line_data.get_line_parameters(actual_wl)
            
            # Print the info message if it exists (e.g. for a dominant line overriding ambiguity)
            if params.get('info_message'):
                print(params['info_message'])

            # Check for NIST ambiguity
            if params.get('is_ambiguous', False):
                print(f"Warning: Multiple NIST lines found within ambiguity window for experimental peak at {actual_wl:.2f} nm. Disregarding.")
                continue

            # Enforce the +/- tolerance check
            diff = abs(params['nist_wl'] - actual_wl)
            if diff <= tolerance:
                match_status = "Matched"
            else:
                match_status = "Unmatched"
                print(f"Warning: Closest NIST line ({params['nist_wl']:.2f} nm) is outside +/- {tolerance} nm tolerance for peak at {actual_wl:.2f} nm.")

            peak_data = {
                'Observed_Wavelength': actual_wl,
                'Integrated_Intensity': intensity,
                'Match_Status': match_status,
                'nist_wavelength': params['nist_wl'] if match_status == 'Matched' else np.nan,
                'A_ki': params['A_ki'] if match_status == 'Matched' else np.nan,
                'Upper_Level_Energy_Ek': params['E_k'] if match_status == 'Matched' else np.nan,
                'n_upper': params['n_upper'] if match_status == 'Matched' else np.nan,
                'g_sub': params['g_sub'] if match_status == 'Matched' else np.nan,
                'g_lumped': params['g_lumped'] if match_status == 'Matched' else np.nan,
                'is_metastable': params['is_metastable'] if match_status == 'Matched' else False,
                'upper_state_id': params['upper_state_id'] if match_status == 'Matched' else np.nan
            }
            detected_peaks.append(peak_data)

        return pd.DataFrame(detected_peaks)

    def calculate_lumped_densities(self, matched_peaks_df, skip_metastables=False):
        """
        Calculates lumped level densities by aggregating measured sublevels 
        using the two-step aggregation approach.
        """
        df = matched_peaks_df[matched_peaks_df['Match_Status'] == 'Matched'].copy()

        if df.empty:
            return pd.DataFrame(), pd.DataFrame(columns=['n_upper', 'lumped_relative_density'])

        if skip_metastables:
            df = df[~df['is_metastable']].copy()
            if df.empty:
                return pd.DataFrame(), pd.DataFrame(columns=['n_upper', 'lumped_relative_density'])

        # Calculate raw density of the specific sublevel (N_k)
        df['density_sublevel'] = df['Integrated_Intensity'] / df['A_ki']

        # Step 1 (Intra-state Deduplication): Group by exact upper state
        # Calculate average raw density if multiple lines share the same state
        state_counts = df.groupby('upper_state_id')['density_sublevel'].transform('count')
        df['averaged_densities'] = np.where(state_counts > 1, 
                                            df.groupby('upper_state_id')['density_sublevel'].transform('mean'), 
                                            np.nan)
        # Calculate relative error (standard deviation / mean) if multiple lines
        # Here we use the unbiased standard deviation (ddof=1)
        def calc_rel_error(x):
            if len(x) > 1:
                std = x.std()
                mean = x.mean()
                return std / mean if mean != 0 else 0
            return np.nan

        df['relative_error'] = np.where(state_counts > 1,
                                         df.groupby('upper_state_id')['density_sublevel'].transform(calc_rel_error),
                                         np.nan)

        # Deduplicate to keep only one entry per physical state
        # We take the first entry but replace its density with the average if applicable
        unique_states_df = df.copy()
        unique_states_df['N_k_final'] = np.where(unique_states_df['averaged_densities'].notna(),
                                                 unique_states_df['averaged_densities'],
                                                 unique_states_df['density_sublevel'])
        
        unique_states_df = unique_states_df.sort_values('Observed_Wavelength').drop_duplicates('upper_state_id')

        # Step 2 (Inter-state Lumping): Group the unique states from Step 1 by their target Vlcek state
        lumped_densities = unique_states_df.groupby('n_upper').agg(
            sum_N_k=('N_k_final', 'sum'),
            sum_g_k=('g_sub', 'sum'),
            g_lumped=('g_lumped', 'first'),
            g_lines_list=('g_sub', lambda x: list(x.astype(str)))
        ).reset_index()

        # Calculate final lumped density: lumped_density = sum_N_k * (g_lump_total / sum_g_k)
        lumped_densities['lumped_relative_density'] = lumped_densities['sum_N_k'] * (lumped_densities['g_lumped'] / lumped_densities['sum_g_k'])
        
        # Format g_lines as a string: "g1 + g2 + ..."
        lumped_densities['g_lines'] = lumped_densities['g_lines_list'].apply(lambda x: ' + '.join(x))

        # Drop intermediate columns
        lumped_densities = lumped_densities.drop(columns=['sum_N_k', 'sum_g_k', 'g_lines_list'])

        return df, lumped_densities