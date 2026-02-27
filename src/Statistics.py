import pandas as pd
from scipy.stats import ttest_rel
import os
from textwrap import dedent


class GlobalStats:
    """
    Generates LaTeX table with statistical analysis of model performance.

    Creates publication-ready tables with paired t-test significance stars comparing
    molecular representations against fingerprint baselines across all experiments.
    """

    def __init__(self):
        """
        Initialize statistics generator and create LaTeX table if not already present.

        Checks for existing 'results/statistics.tex' file. If missing, processes
        all experiment results to generate table with MAE±SD values and significance
        stars from paired t-tests.
        """

        self.output_path = 'results/statistics.tex'
        if os.path.exists(self.output_path):
            return

        print("Creating paired t-test statistics in LaTeX")

        self.models = {'LR': 'singleDense', 'MLP': 'simpleModel'}
        self.reps = ['fingerprints', 'unimol', 'chemberta', 'molformer', 'molmim']
        self.base_path = 'results'

        self.exp_labels = [
            'CCSBase--CCSBase',
            'CCSBase--METLINCCS',
            'METLINCCS--CCSBase',
            'METLINCCS--METLINCCS',
            r'$\mu_{\text{eff}}$--$\mu_{\text{eff}}$',
            'SMRT--SMRT'
        ]

        self.exp_files = [
            'train_val_ccsbase_test_ccsbase_results.csv',
            'train_val_ccsbase_test_metlinccs_results.csv',
            'train_val_metlinccs_test_ccsbase_results.csv',
            'train_val_metlinccs_test_metlinccs_results.csv',
            'train_val_mobility_test_mobility_results.csv',
            'train_val_smrt_test_smrt_results.csv'
        ]

        self.latex = self._build_table_header()
        self._process_experiments()
        self.latex += dedent('''\
            \\bottomrule
            \\end{tabular}
            \\end{table}
            \\end{landscape}
            \\end{document}
        ''')
        self._save_to_file()

    @staticmethod
    def _build_table_header():
        """
        Build LaTeX document header with table formatting.

        Returns:
            str: Complete LaTeX preamble and table header with multi-column formatting
                 for different molecular representations and model architectures.
        """

        return dedent('''\
            \\documentclass[unnumsec,webpdf,contemporary,large]{oup-authoring-template}%
            \\usepackage{booktabs}
            \\usepackage{lscape}
            \\begin{document}
            \\begin{landscape}
            \\begin{table}[H]
            \\centering
            \\small
            \\caption{MAE±SD for all models and tasks.}
            \\label{tab:results_all}
            \\setlength{\\tabcolsep}{4pt}
            \\begin{tabular}{lcccccccccc}
            \\toprule
            & \\multicolumn{2}{c}{\\textbf{Fingerprints}} &
              \\multicolumn{2}{c}{\\textbf{UniMol}} &
              \\multicolumn{2}{c}{\\textbf{ChemBERTa}} &
              \\multicolumn{2}{c}{\\textbf{MolFormer}} &
              \\multicolumn{2}{c}{\\textbf{MolMIM}} \\\\
            \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}
            \\textbf{Train--Test} &
            \\textbf{LR} & \\textbf{MLP} &
            \\textbf{LR} & \\textbf{MLP} &
            \\textbf{LR} & \\textbf{MLP} &
            \\textbf{LR} & \\textbf{MLP} &
            \\textbf{LR} & \\textbf{MLP} \\\\
            \\midrule
        ''')

    @staticmethod
    def _get_stars(p):
        """
        Convert p-value to statistical significance stars.

        Args:
            p (float): P-value from statistical test

        Returns:
            str: Significance stars: *** for p<0.001, ** for p<0.01, * for p<0.05,
                 empty string otherwise
        """

        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return ''

    @staticmethod
    def _parse_mean_std(value):
        """
        Parse mean±std formatted string into numeric values.

        Args:
            value (str): String in format "mean±std"

        Returns:
            tuple: (mean, std) as float values
        """

        mean, std = value.split('±')
        return float(mean), float(std)

    @staticmethod
    def _read_csv(path):
        """
        Read results CSV file and separate fold-wise from total metrics.

        Args:
            path (str): Path to results CSV file

        Returns:
            tuple: (mae_folds, total_mae) where:
                   - mae_folds: numpy array of MAE values per fold
                   - total_mae: tuple of (mean, std) from 'Total' row
        """

        df = pd.read_csv(path)

        folds = df[pd.to_numeric(df['Fold'], errors='coerce').notna()].copy()
        folds['MAE'] = pd.to_numeric(folds['MAE'], errors='raise')

        total = df[df['Fold'] == 'Total'].iloc[0]
        mean, std = GlobalStats._parse_mean_std(total['MAE'])

        return folds['MAE'].to_numpy(dtype=float), (mean, std)

    def _load_experiment(self, exp_file):
        """
        Load all model results for a specific experiment.

        Args:
            exp_file (str): Name of results CSV file

        Returns:
            tuple: (folds, totals) dictionaries with keys (model, representation)
        """

        folds = {}
        totals = {}

        for model, model_dir in self.models.items():
            for rep in self.reps:
                path = os.path.join(self.base_path, model_dir, rep, exp_file)
                mae_folds, total_mae = self._read_csv(path)
                folds[(model, rep)] = mae_folds
                totals[(model, rep)] = total_mae

        return folds, totals

    def _process_experiments(self):
        """
        Process all experiments and build LaTeX table rows.

        Iterates through experiment labels and files, loads results for each,
        and appends formatted LaTeX rows to the table content.
        """

        for i, (label, file) in enumerate(zip(self.exp_labels, self.exp_files)):
            folds, totals = self._load_experiment(file)
            self.latex += self._build_rows(label, folds, totals, i == len(self.exp_labels) - 1)

    def _build_rows(self, label, folds, totals, is_last):
        """
        Build LaTeX rows for a single experiment.

        Args:
            label (str): Experiment label for table row
            folds (dict): Fold-wise MAE arrays for all model/rep combinations
            totals (dict): Mean±std tuples for all model/rep combinations
            is_last (bool): Whether this is the final experiment row

        Returns:
            str: LaTeX code containing value row and significance stars row
        """

        # Values formatting
        values = []
        for rep in self.reps:
            for model in ['LR', 'MLP']:
                mean, std = totals[(model, rep)]  # full precision floats
                values.append(f"{mean:.2f}±{std:.2f}")

        value_row = label + ' & ' + ' & '.join(values) + r' \\'

        # Paired t-test
        baseline = {
            'LR': folds[('LR', 'fingerprints')],
            'MLP': folds[('MLP', 'fingerprints')]
        }

        stars = [' ', ' ', ' ']  # empty cells for first three columns
        for rep in self.reps[1:]:
            for model in ['LR', 'MLP']:
                _, p = ttest_rel(
                    baseline[model],
                    folds[(model, rep)]
                )
                stars.append(self._get_stars(p))

        stars_row = ' & '.join(stars) + r' \\'
        if not is_last:
            stars_row += '[4pt]'

        return value_row + '\n' + stars_row + '\n'

    def _save_to_file(self):
        """
        Save complete LaTeX table to output file.

        Creates results directory if needed and writes the final LaTeX document
        to 'results/statistics.tex'.
        """

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as f:
            f.write(self.latex)
