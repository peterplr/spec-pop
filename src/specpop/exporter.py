import pandas as pd
import os

class Exporter:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_results(self, results, result_filename='intensities.csv'):
        df_results = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, result_filename)
        df_results.to_csv(csv_path, index=False)
        print(f"\nAll lines analyzed! Summary saved to {csv_path}")

    def save_aggregated_results(self, aggregated_results, filename='aggregated_densities.csv'):
        df_agg = pd.DataFrame(aggregated_results)
        csv_path = os.path.join(self.output_dir, filename)
        df_agg.to_csv(csv_path, index=False)
        print(f"Aggregated number densities saved to {csv_path}")
