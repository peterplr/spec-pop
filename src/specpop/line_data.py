import pandas as pd
import numpy as np
import os
import re

# The directory where this script is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class LineData:
    def __init__(self, nist_path=None, vlcek_path=None, nist_ambiguity_window=1.0):
        if nist_path is None:
            nist_path = os.path.join(_SCRIPT_DIR, 'data', 'NIST', 'nist_argon_raw.csv')
        if vlcek_path is None:
            vlcek_path = os.path.join(_SCRIPT_DIR, 'data', 'Vlcek', 'excitation_levels.csv')

        self.nist_data = pd.read_csv(nist_path)
        self.nist_ambiguity_window = nist_ambiguity_window
        # Clean up column names and convert relevant columns
        self.nist_data.columns = self.nist_data.columns.str.strip().str.replace('"', '')
        self.nist_data['obs_wl_vac(nm)'] = self.nist_data['obs_wl_vac(nm)'].str.replace('=', '').str.replace('"', '').astype(float)
        self.nist_data['Aki(s^-1)'] = self.nist_data['Aki(s^-1)'].str.replace('=', '').str.replace('"', '').astype(float)
        self.nist_data['Ek(eV)'] = self.nist_data['Ek(eV)'].str.replace('=', '').str.replace('"', '').astype(float)
        self.nist_data['Ei(eV)'] = self.nist_data['Ei(eV)'].str.replace('=', '').str.replace('"', '').astype(float)
        self.nist_data['g_k'] = self.nist_data['g_k'].astype(int)
        self.nist_data['Rel. Int.'] = pd.to_numeric(self.nist_data['intens'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
        self.nist_data['conf_k'] = self.nist_data['conf_k'].astype(str).str.replace('=', '').str.replace('"','').str.strip()
        self.nist_data['term_k'] = self.nist_data['term_k'].astype(str).str.replace('=', '').str.replace('"','').str.strip()
        self.nist_data['J_k'] = self.nist_data['J_k'].astype(str).str.replace('=', '').str.replace('"', '').str.strip()

        self.vlcek_data = pd.read_csv(vlcek_path)
        # Clean up column names and convert relevant columns
        self.vlcek_data.columns = self.vlcek_data.columns.str.strip()
        self.vlcek_data['excitation_energy'] = self.vlcek_data['excitation_energy'].astype(float)
        self.vlcek_data['n'] = self.vlcek_data['n'].astype(int)
        self.vlcek_data['g'] = self.vlcek_data['g'].astype(int)

    def _parse_nist_to_qns(self, conf, term, j_str):
        if "3p6" in conf and "1S" in term and j_str == "0":
            return {"n": 3, "l": 1, "K": "all", "J": 0, "core": "3/2"}

        shell = conf.split('.')[-1]
        match_n = re.search(r'(\d+)', shell)
        match_l = re.search(r'([spdfghi])', shell.lower())

        if not match_n or not match_l:
            return None

        n_val = int(match_n.group(1))
        l_val = "spdfghi".find(match_l.group(1))
        core = "1/2" if "<1/2>" in conf else "3/2"
        match_k = re.search(r'\[(.*?)]', term)
        k_val = match_k.group(1) if match_k else "all"
        j_val = int(j_str) if j_str.isdigit() else "all"

        return {"n": n_val, "l": l_val, "K": k_val, "J": j_val, "core": core}

    def get_line_parameters(self, wavelength):
        # Check for ambiguity in NIST data
        search_radius = self.nist_ambiguity_window / 2.0
        search_min = wavelength - search_radius
        search_max = wavelength + search_radius

        nearby_lines = self.nist_data[
            (self.nist_data['obs_wl_vac(nm)'] >= search_min) &
            (self.nist_data['obs_wl_vac(nm)'] <= search_max)
            ]

        is_ambiguous = False
        info_message = None
        
        if len(nearby_lines) > 1:
            if 'Rel. Int.' in self.nist_data.columns:
                # Get the maximum intensity among the nearby lines
                max_intensity = nearby_lines['Rel. Int.'].max()
                
                # Check if the max intensity is at least 100x larger than ALL other nearby lines
                # that have non-zero intensity.
                other_lines = nearby_lines[nearby_lines['Rel. Int.'] < max_intensity]
                
                if not other_lines.empty and max_intensity >= 100 * other_lines['Rel. Int.'].max():
                    # We have a dominant line
                    is_ambiguous = False
                    dominant_wl = nearby_lines.loc[nearby_lines['Rel. Int.'].idxmax(), 'obs_wl_vac(nm)']
                    info_message = f"Info: Multiple NIST lines near {wavelength:.2f} nm, but line at {dominant_wl:.2f} nm is dominant (rel. int. > 100x higher). Disregarding ambiguity."
                else:
                    is_ambiguous = True
            else:
                # If relative intensity data is missing, we must be conservative and flag as ambiguous
                is_ambiguous = True

        # Find the single closest wavelength in NIST data (or the dominant one if we resolved ambiguity)
        if not is_ambiguous and len(nearby_lines) > 1 and 'Rel. Int.' in self.nist_data.columns:
             nist_idx = nearby_lines['Rel. Int.'].idxmax()
             nist_row = self.nist_data.loc[nist_idx]
        else:
             nist_idx = (np.abs(self.nist_data['obs_wl_vac(nm)'] - wavelength)).argmin()
             nist_row = self.nist_data.iloc[nist_idx]

        A_ki = nist_row['Aki(s^-1)']
        g_sub = nist_row['g_k']
        ek_ev = nist_row['Ek(eV)']
        ei_ev = nist_row['Ei(eV)']

        is_meta = bool(abs(ei_ev - 11.548) < 0.05 or abs(ei_ev - 11.723) < 0.05)

        conf_k = nist_row['conf_k']
        term_k = nist_row['term_k']
        j_k = nist_row['J_k']

        nist_qns = self._parse_nist_to_qns(conf_k, term_k, j_k)

        if nist_qns is None:
            # Fallback if NIST designation is completely unparseable
            vlcek_idx = (np.abs(self.vlcek_data['excitation_energy'] - ek_ev)).argmin()
            vlcek_row = self.vlcek_data.iloc[vlcek_idx]
        else:
            # Helper to handle lumped Vlcek columns safely
            def check_match(vlcek_val, nist_val):
                # Split ONLY by slashes surrounded by spaces to preserve fractions like '3/2'
                valid_vals = [v.strip() for v in re.split(r'\s+/\s+', str(vlcek_val))]
                return 'all' in valid_vals or str(nist_val) in valid_vals

            # Filter Vlcek dataframe based on matching quantum numbers
            def qn_match(row):
                match_n = check_match(row.get('qn_n', ''), nist_qns['n'])
                match_l = check_match(row.get('qn_l', ''), nist_qns['l'])
                match_c = check_match(row.get('qn_core', ''), nist_qns['core'])
                match_k = check_match(row.get('qn_K', ''), nist_qns['K'])
                match_j = check_match(row.get('qn_J', ''), nist_qns['J'])

                return match_n and match_l and match_c and match_k and match_j

            valid_mask = self.vlcek_data.apply(qn_match, axis=1)
            filtered_vlcek = self.vlcek_data[valid_mask]

            if not filtered_vlcek.empty:
                # Use energy ONLY to break the tie among the EXACT designation matches.
                vlcek_idx = (np.abs(filtered_vlcek['excitation_energy'] - ek_ev)).argmin()
                vlcek_row = filtered_vlcek.iloc[vlcek_idx]
            else:
                # Absolute fallback if no designation matches exist
                print(
                    f"  [!] WARNING: No Vlcek match found for NIST {conf_k} {term_k} J={j_k}. Falling back to energy.")
                vlcek_idx = (np.abs(self.vlcek_data['excitation_energy'] - ek_ev)).argmin()
                vlcek_row = self.vlcek_data.iloc[vlcek_idx]

        # n_upper uses the Vlcek index 'n' (1-39), not the principal quantum number
        n_upper = vlcek_row['n']
        g_lumped = vlcek_row['g']

        return {
            'center': wavelength,
            'nist_wl': nist_row['obs_wl_vac(nm)'],
            'E_k': ek_ev,
            'A_ki': A_ki,
            'n_upper': n_upper,
            'g_sub': g_sub,
            'g_lumped': g_lumped,
            'is_metastable': is_meta,
            'is_ambiguous': is_ambiguous,
            'info_message': info_message
        }