import pandas as pd
import numpy as np
import os

# The directory where this script is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The project root is two levels up (src/oes_lines -> src -> root)
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))

class LineData:
    def __init__(self, nist_path=None, vlcek_path=None, nist_ambiguity_window=1.0):
        if nist_path is None:
            nist_path = os.path.join(_PROJECT_ROOT, 'data', 'NIST', 'nist_argon_raw.csv')
        if vlcek_path is None:
            vlcek_path = os.path.join(_PROJECT_ROOT, 'data', 'Vlcek', 'excitation_levels.csv')

        self.nist_data = pd.read_csv(nist_path)
        self.nist_ambiguity_window = nist_ambiguity_window
        # Clean up column names and convert relevant columns
        self.nist_data.columns = self.nist_data.columns.str.strip().str.replace('"', '')
        self.nist_data['obs_wl_air(nm)'] = self.nist_data['obs_wl_air(nm)'].str.replace('=', '').str.replace('"', '').astype(float)
        self.nist_data['Aki(s^-1)'] = self.nist_data['Aki(s^-1)'].str.replace('=', '').str.replace('"', '').astype(float)
        self.nist_data['Ek(eV)'] = self.nist_data['Ek(eV)'].str.replace('=', '').str.replace('"', '').astype(float)
        self.nist_data['Ei(eV)'] = self.nist_data['Ei(eV)'].str.replace('=', '').str.replace('"', '').astype(float)
        self.nist_data['g_k'] = self.nist_data['g_k'].astype(int)

        self.vlcek_data = pd.read_csv(vlcek_path)
        # Clean up column names and convert relevant columns
        self.vlcek_data.columns = self.vlcek_data.columns.str.strip()
        self.vlcek_data['excitation_energy'] = self.vlcek_data['excitation_energy'].astype(float)
        self.vlcek_data['n'] = self.vlcek_data['n'].astype(int)
        self.vlcek_data['g'] = self.vlcek_data['g'].astype(int)

    def get_line_parameters(self, wavelength):
        # Check for ambiguity in NIST data
        search_radius = self.nist_ambiguity_window / 2.0
        search_min = wavelength - search_radius
        search_max = wavelength + search_radius

        nearby_lines = self.nist_data[
            (self.nist_data['obs_wl_air(nm)'] >= search_min) &
            (self.nist_data['obs_wl_air(nm)'] <= search_max)
            ]

        is_ambiguous = len(nearby_lines) > 1

        # Find the single closest wavelength in NIST data
        nist_idx = (np.abs(self.nist_data['obs_wl_air(nm)'] - wavelength)).argmin()
        nist_row = self.nist_data.iloc[nist_idx]

        A_ki = nist_row['Aki(s^-1)']
        g_sub = nist_row['g_k']
        ek_ev = nist_row['Ek(eV)']
        ei_ev = nist_row['Ei(eV)']

        is_meta = bool(abs(ei_ev - 11.548) < 0.05 or abs(ei_ev - 11.723) < 0.05)

        # --- NEW QUANTUM NUMBER MAPPING LOGIC ---

        # 1. Calculate J from NIST g_k (for neutral Argon, J is always an integer)
        # Formula: g = 2J + 1  =>  J = (g - 1) / 2
        nist_j_str = str(int((g_sub - 1) / 2))

        # 2. Filter the Vlcek data based on the J quantum number
        def j_matches(qn_j_val):
            # Split the Vlcek string by '/' and strip whitespace
            valid_js = [j.strip() for j in str(qn_j_val).split('/')]
            # Match if the NIST J is in the list, or if the level applies to 'all' J's
            return (nist_j_str in valid_js) or ('all' in valid_js)

        valid_mask = self.vlcek_data['qn_J'].apply(j_matches)
        filtered_vlcek = self.vlcek_data[valid_mask]

        # 3. Find the closest energy ONLY among the levels with the correct J
        if not filtered_vlcek.empty:
            vlcek_idx = (np.abs(filtered_vlcek['excitation_energy'] - ek_ev)).argmin()
            vlcek_row = filtered_vlcek.iloc[vlcek_idx]
        else:
            # Fallback to pure energy mapping just in case a high-lying level is missing a J value
            vlcek_idx = (np.abs(self.vlcek_data['excitation_energy'] - ek_ev)).argmin()
            vlcek_row = self.vlcek_data.iloc[vlcek_idx]

        n_upper = vlcek_row['n']
        g_lumped = vlcek_row['g']

        return {
            'center': wavelength,
            'nist_wl': nist_row['obs_wl_air(nm)'],
            'E_k': ek_ev,
            'A_ki': A_ki,
            'n_upper': n_upper,
            'g_sub': g_sub,
            'g_lumped': g_lumped,
            'is_metastable': is_meta,
            'is_ambiguous': is_ambiguous
        }