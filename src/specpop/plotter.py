import matplotlib.pyplot as plt
import os
import numpy as np

# The directory where this script is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The project root is two levels up (src/oes_lines -> src -> root)
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
_STYLE_PATH = os.path.join(_PROJECT_ROOT, 'configs', 'custom.mplstyle')

class Plotter:
    def __init__(self, output_dir='output', plot_margin=20.0):
        self.output_dir = output_dir
        self.plot_margin = plot_margin
        os.makedirs(self.output_dir, exist_ok=True)
        # Apply the custom style sheet for all plots created by this class
        plt.style.use(_STYLE_PATH)

    def plot_line_integration(self, wl_win, counts_win, baseline, center, intensity):
        # Removed explicit figsize to use the one from custom.mplstyle (figure.figsize: 9, 6)
        fig, ax = plt.subplots() 
        ax.plot(wl_win, counts_win, 'ko-', label='Raw Data', markersize=4)
        ax.plot(wl_win, baseline, 'r--', label='Linear Baseline')
        ax.fill_between(wl_win, baseline, counts_win,
                         where=(counts_win > baseline),
                         interpolate=True, color='#383a6b',
                         label=f'Area: {intensity:.2e}',
                         alpha=0.5)
        ax.set_title(f"Integration of {center:.2f} nm Line")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Counts")
        ax.legend()
        
        plot_path = os.path.join(self.output_dir, f"line_{center:.2f}nm.png")
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Analyzed {center:.2f} nm -> Area: {intensity:.2e}. Plot saved.")

    def plot_overview(self, wavelengths, counts, analyzed_lines):

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(wavelengths, counts, marker='', linewidth=0.8, label='Spectrum')

        all_wl = [line['Actual_Wavelength_nm'] for line in analyzed_lines]
        
        if not all_wl: # Handle case where no lines were analyzed
            ax.set_title("Overview of Spectral Lines (No lines analyzed)")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Counts")
            ax.legend()
            plot_path = os.path.join(self.output_dir, "overview.png")
            fig.savefig(plot_path)
            plt.close(fig)
            print(f"Overview plot saved to {plot_path}")
            return

        min_wl, max_wl = min(all_wl), max(all_wl)
        
        x_min_lim = min_wl - self.plot_margin
        x_max_lim = max_wl + self.plot_margin
        ax.set_xlim(x_min_lim, x_max_lim)

        visible_mask = (wavelengths >= x_min_lim) & (wavelengths <= x_max_lim)
        visible_counts = counts[visible_mask]
        
        # Adjust y-limits for logarithmic scale
        min_visible_counts_positive = np.min(visible_counts[visible_counts > 0]) if np.any(visible_counts > 0) else 1
        y_min_lim = max(1, min_visible_counts_positive * 0.5)
        y_max_lim = np.max(visible_counts) * 2.0 # More headroom for log scale
        ax.set_ylim(bottom=y_min_lim, top=y_max_lim)

        for line in sorted(analyzed_lines, key=lambda x: x['Actual_Wavelength_nm']):
            center = line['Actual_Wavelength_nm']
            width = line['width']
            left_bound = center - (width / 2.0)

            # Add colored box with new color
            ax.add_patch(
                plt.Rectangle((left_bound, y_min_lim), width, y_max_lim - y_min_lim,
                              facecolor='#cb1f73', alpha=0.1,
                              linewidth=0.8, linestyle='-', edgecolor='#cb1f73',
                              label='Integration Window' if 'Integration Window' not in [h.get_label() for h in ax.get_legend_handles_labels()[0]] else '')
            )

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Counts")

        ax.legend()

        plot_path = os.path.join(self.output_dir, "overview.png")
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Overview plot saved to {plot_path}")
