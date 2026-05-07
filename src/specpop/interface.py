import tomllib
import os
import pandas as pd
from .analysis import OESAnalysis
from .line_data import LineData
from .plotter import Plotter
from .exporter import Exporter


class Interface:
    def __init__(self, config_path):
        # Get the directory where the config file is located
        config_dir = os.path.dirname(os.path.abspath(config_path))

        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

        # --- Resolve paths relative to the config file location ---
        data_path = os.path.join(config_dir, self.config['data_path'])

        dark_image_path = self.config.get('dark_image_path')
        if dark_image_path:
            dark_image_path = os.path.join(config_dir, dark_image_path)

        output_dir = os.path.join(config_dir, self.config.get('output_dir', 'output'))

        # --- Get settings with defaults ---
        peak_integration = self.config.get('peak_integration', {})
        auto_peak = self.config.get('auto_peak', {})
        plotting_config = self.config.get('plotting', {})
        parser_settings = self.config.get('parser', {})
        
        # Fallback to general settings if subcategories are not present for backwards compatibility
        settings = self.config.get('settings', {})

        self.fwhm_multiplier = peak_integration.get('fwhm_multiplier', settings.get('fwhm_multiplier', 3.0))
        self.fwhm_search_window = peak_integration.get('fwhm_search_window', settings.get('fwhm_search_window', 10.0))
        self.default_width = peak_integration.get('default_width', settings.get('default_width', 1.0))
        
        self.plot_margin = plotting_config.get('plot_margin', settings.get('plot_margin', 20.0))
        self.plot_overview_enabled = plotting_config.get('plot_overview', settings.get('plot_overview', True))
        self.plot_separate_lines = plotting_config.get('plot_separate_lines', settings.get('plot_separate_lines', True))
        
        self.min_auto_peak_height = auto_peak.get('min_auto_peak_height', settings.get('min_auto_peak_height', 500))
        self.auto_peak_detection_range_min = auto_peak.get('auto_peak_detection_range_min', settings.get('auto_peak_detection_range_min'))
        self.auto_peak_detection_range_max = auto_peak.get('auto_peak_detection_range_max', settings.get('auto_peak_detection_range_max'))
        self.match_tolerance = auto_peak.get('match_tolerance', settings.get('match_tolerance', 1.0))
        self.skip_metastables = auto_peak.get('skip_metastables', settings.get('skip_metastables', False))
        self.nist_ambiguity_window = auto_peak.get('nist_ambiguity_window', settings.get('nist_ambiguity_window', 1.0))
        self.ignore_wavelengths = auto_peak.get('ignore_wavelengths', settings.get('ignore_wavelengths', []))

        self.wl_col_name = parser_settings.get('wl_col_name', 'Wavelength')
        self.count_col_name = parser_settings.get('count_col_name', 'Counts')
        self.delimiter = parser_settings.get('delimiter', '\t')

        # --- Initialize Core Components ---
        # 1. Initialize LineData first
        self.line_data = LineData(nist_ambiguity_window=self.nist_ambiguity_window)

        # 2. Inject LineData into OESAnalysis
        self.analyzer = OESAnalysis(
            data_path=data_path,
            line_data_provider=self.line_data,
            dark_image_path=dark_image_path,
            wl_col_name=self.wl_col_name,
            count_col_name=self.count_col_name,
            delimiter=self.delimiter,
            fwhm_multiplier=self.fwhm_multiplier,
            fwhm_search_window=self.fwhm_search_window, # Pass the correct parameter
            default_width=self.default_width,
            ignore_wavelengths=self.ignore_wavelengths
        )

        self.plotting = Plotter(output_dir=output_dir, plot_margin=self.plot_margin)
        self.exporter = Exporter(output_dir=output_dir)

    def run_analysis(self):
        wavelength_configs = self.config.get('wavelengths', [])
        analyzed_lines_for_overview = []

        if not wavelength_configs:
            print("No specific wavelengths provided. Auto-detecting peaks and matching to NIST...")

            # --- AUTO-DETECTION MODE ---
            matched_peaks_df = self.analyzer.auto_find_and_match(
                min_height=self.min_auto_peak_height,
                range_min=self.auto_peak_detection_range_min,
                range_max=self.auto_peak_detection_range_max,
                tolerance=self.match_tolerance
            )

            if matched_peaks_df.empty:
                print("No peaks detected above the minimum height or within the specified range. Exiting.")
                return

            # Calculate the lumped densities using the new DataFrame method
            sublevel_df, lumped_df = self.analyzer.calculate_lumped_densities(
                matched_peaks_df,
                skip_metastables=self.skip_metastables  # ADD ARGUMENT HERE
            )

            # Quick loop to generate individual integration plots for matched lines
            for _, row in sublevel_df.iterrows():
                wl = row['Observed_Wavelength']
                # Re-fetch the window arrays specifically for plotting
                intensity, wl_win, counts_win, baseline, width, actual_wl, peak_height = \
                    self.analyzer.calculate_line_intensity(wl)

                if self.plot_separate_lines:
                    self.plotting.plot_line_integration(wl_win, counts_win, baseline, actual_wl, intensity)

                # Added 'width' here to prevent the KeyError in plot_overview
                analyzed_lines_for_overview.append({
                    'Actual_Wavelength_nm': actual_wl,
                    'Total_Counts': intensity,
                    'width': width
                })

            # Export DataFrames (converting to list of dicts to match standard exporter expectations)
            self.exporter.save_results(sublevel_df.to_dict(orient='records'))
            self.exporter.save_aggregated_results(lumped_df.to_dict(orient='records'))

        else:
            print("Specific wavelengths provided. Processing target list...")

            # --- MANUAL LIST MODE ---
            sublevel_df = self.analyzer.process_target_list(wavelength_configs)

            if sublevel_df.empty:
                print("No specific wavelengths processed. Exiting.")
                return

            # Calculate the lumped densities using the new DataFrame method
            _, lumped_df = self.analyzer.calculate_lumped_densities(
                sublevel_df,
                skip_metastables=self.skip_metastables
            )

            # Quick loop to generate individual integration plots for matched lines
            for idx, row in sublevel_df.iterrows():
                wl = row['Observed_Wavelength']
                target_wl = row['Target_Wavelength']
                # Try to get the original manual width from the config based on the target_wl
                original_config = next((item for item in wavelength_configs if item['center'] == target_wl), None)
                manual_width = original_config.get('width') if original_config else None

                # Re-fetch the window arrays specifically for plotting
                intensity, wl_win, counts_win, baseline, width, actual_wl, peak_height = \
                    self.analyzer.calculate_line_intensity(target_wl, manual_width=manual_width)

                if self.plot_separate_lines:
                    self.plotting.plot_line_integration(wl_win, counts_win, baseline, actual_wl, intensity)

                # Added 'width' here to prevent the KeyError in plot_overview
                analyzed_lines_for_overview.append({
                    'Actual_Wavelength_nm': actual_wl,
                    'Total_Counts': intensity,
                    'width': width
                })

            # Export manual results
            self.exporter.save_results(sublevel_df.to_dict(orient='records'))
            self.exporter.save_aggregated_results(lumped_df[['n_upper', 'Lumped_Relative_Density', 'Lumped_Relative_Error']].to_dict(orient='records'))

        # Plot the final global overview
        if self.plot_overview_enabled:
            self.plotting.plot_overview(self.analyzer.wavelengths, self.analyzer.counts, analyzed_lines_for_overview)


if __name__ == '__main__':
    interface = Interface(os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.toml'))
    interface.run_analysis()