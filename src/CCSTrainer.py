import csv
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.model_selection import GroupKFold
import warnings

from src.CCSTrainer_extra.data_handling import load_data, handle_duplicates
from src.CCSTrainer_extra.train_model_functions import train_a_fold

warnings.filterwarnings('ignore')


class CCSTrainer:
    """
    Trainer class for CCS (Collision Cross Section) prediction models.

    Handles training of models across different molecular representations,
    dataset configurations, and neural network architectures. Automatically
    checks if training has been completed to avoid redundant computation.
    """

    def __init__(self):
        """
        Initialize the CCS trainer and start training if not already completed.

        Checks for existing results directories and if not found, initiates
        training across all model architectures and molecular representations.
        """

        # Return if done
        if os.path.exists("results/singleDense") and os.path.exists("results/simpleModel"):
            return
        else:
            print("Starting CCStrainer:")

        for nn in ["singleDense", "simpleModel"]:
            for file in ["all_datasets_unimol_unified.csv", "all_datasets_chemberta_unified.csv", "all_datasets_molformer_unified.csv",
                         "all_datasets_molmim_unified.csv", "all_datasets_fingerprints_unified.csv"]:
                self.training(file, nn)


    def training(self, file, nn):
        """
        Execute training pipeline for a specific representation file and neural network.

        Args:
            file (str): Unified CSV file containing molecular representation data
            nn (str): Neural network architecture name ('singleDense' or 'simpleModel')

        Processes all dataset configurations, applies dataset-specific filters,
        performs k-fold splitting, trains models, and saves results.
        """

        # Parameters
        subfolder1 = nn

        folders = {
            "resources": "resources",
            "results"  : "results"
        }

        # Dataset configurations
        dataset_configs = [
            {"train_val": ["ccsbase"], "test": "ccsbase"},
            {"train_val": ["ccsbase"], "test": "metlinccs"},
            {"train_val": ["metlinccs"], "test": "ccsbase"},
            {"train_val": ["metlinccs"], "test": "metlinccs"},
            {"train_val": ["mobility"], "test": "mobility"},
            {"train_val": ["smrt"], "test": "smrt"},
        ]

        # Determine features and subfolder1 name based on the file name
        if "unimol" in file:
            subfolder2 = "unimol"
            features = [f"unimol_{i}" for i in range(1536)]
        elif "chemberta" in file:
            subfolder2 = "chemberta"
            features = [f"chemberta_{i}" for i in range(768)]
        elif "molformer" in file:
            subfolder2 = "molformer"
            features = [f"molformer_{i}" for i in range(768)]
        elif "molmim" in file:
            subfolder2 = "molmim"
            features = [f"molmim_{i}" for i in range(512)]
        elif "fingerprints" in file:
            subfolder2 = "fingerprints"
            features = [f"V{i + 1}" for i in range(2214)]
        else:
            raise ValueError(f"Unknown file: {file}")

        columns = {
            "dataset": "dataset",
            "id": "inchi",
            "features": features,
            "adduct": "adduct",
            "target": "ccs"
        }

        # Train with a given config of train_val / test
        for config in dataset_configs:
            train_val_dbs = config["train_val"]
            test_db = config["test"]
            print(f"  - train_val: {train_val_dbs}, test: {test_db}")
            csv_filepath = os.path.join(folders["resources"], file)
            train_val_test_df = self._process_dataset(csv_filepath, train_val_dbs, test_db, columns)
            # Filters
            if config["train_val"][0] == "smrt":    # Remove non-retained from SMRT
                target_column = columns['target']
                train_val_test_df = train_val_test_df[train_val_test_df[target_column] >= 300].reset_index(drop=True)
            elif config["train_val"][0] == "mobility":  # Remove large (in absolute value) effective mobilities
                target_column = columns['target']
                train_val_test_df = train_val_test_df[abs(train_val_test_df[target_column]) <= 4500].reset_index(drop=True)
            train_dfs, val_dfs, test_dfs = self._kfold_split(train_val_dbs, test_db, train_val_test_df, folders, columns)
            # Name of the results folder inside results
            subfolder3 = os.path.join(f"train_val_{'_'.join(train_val_dbs)}_test_{test_db}")
            fold_dir = os.path.join(folders["results"], subfolder1, subfolder2, subfolder3)
            # Train a 5-fold cross validation
            for fold, (train_df, val_df, test_df) in enumerate(zip(train_dfs, val_dfs, test_dfs)):
                print(f"  - Training fold {fold + 1}")
                # Train the model with the data for one of the folds
                self._train_model(train_df, val_df, test_df, fold_dir, fold, columns, nn)
                # break (Only for short test, it performs one fold and stop)
            self._write_final_metrics(partial_path=fold_dir)


    @staticmethod
    def _process_dataset(csv_filepath, train_val_dbs, test_db, columns):
        """
        Load and preprocess dataset for a specific train-validation-test configuration.

        Args:
            csv_filepath (str): Path to the unified CSV file
            train_val_dbs (list): List of dataset names for training and validation
            test_db (str): Dataset name for testing
            columns (dict): Column mapping configuration

        Returns:
            pandas.DataFrame: Processed DataFrame with duplicates removed based on
                            CCS threshold of 3% difference
        """

        # Threshold for duplicates,
        ccs_threshold = 0.03

        # If in the config we selected the same database for training and testing, then we load one single database,
        # that will be split into: train, val and test
        if train_val_dbs[0] == test_db and len(train_val_dbs) == 1:
            train_val_test_db = train_val_dbs[0]
            train_val_test_df = load_data(csv_filepath=csv_filepath, dataset_name=train_val_test_db, columns=columns)
        else:  # Otherwise we must load at least two databases, one or more for train_val and one for test
            train_val_df = None
            for train_val_db in train_val_dbs:
                df = load_data(csv_filepath=csv_filepath, dataset_name=train_val_db, columns=columns)
                train_val_df = df if train_val_df is None else pd.concat([train_val_df, df], ignore_index=True)
            test_df = load_data(csv_filepath=csv_filepath, dataset_name=test_db, columns=columns)
            train_val_test_df = pd.concat([train_val_df, test_df])

        # Remove duplicates that differ in their ccs more than a 'ccs_threshold' per one
        print(f"  - Removing duplicates which differ more than a {round(ccs_threshold*100)} per cent")
        train_val_test_df = handle_duplicates(
            train_val_test_df=train_val_test_df,
            columns=columns,
            ccs_threshold=ccs_threshold,
            train_val_dbs=train_val_dbs
        )
        return train_val_test_df

    @staticmethod
    def _kfold_split(train_val_dbs, test_db, train_val_test_df, folders, columns):
        """
        Split data into k-folds for cross-validation.

        Args:
            train_val_dbs (list): List of dataset names for training/validation
            test_db (str): Dataset name for testing
            train_val_test_df (pandas.DataFrame): Combined DataFrame to split
            folders (dict): Directory paths for resources and results
            columns (dict): Column mapping configuration

        Returns:
            tuple: Three lists containing DataFrames for train, validation, and test splits
        """

        # Initialize K-fold splitter
        n_splits = 5  # Number of folds
        group_kfold = GroupKFold(n_splits=n_splits)

        # Initialize lists to store train, validation, and test splits
        train_dfs = []
        val_dfs = []
        test_dfs = []
        # Initialize the fold indices and the list of records
        fold_idx = 1
        records = []
        if train_val_dbs[0] == test_db and len(train_val_dbs) == 1:
            # Split by molecule (columns["id"]) when using same database
            groups = train_val_test_df[columns["id"]]
            for train_val_idx, test_idx in group_kfold.split(train_val_test_df, y=None, groups=groups):
                train_val_df = train_val_test_df.iloc[train_val_idx].copy()
                test_df = train_val_test_df.iloc[test_idx].copy()

                # Split train-val subset into train and val
                train_val_groups = groups[train_val_idx].reset_index(drop=True)
                group_kfold_inner = GroupKFold(n_splits=5)  # e.g., 5 for 80-20
                train_idx, val_idx = next(group_kfold_inner.split(train_val_df, y=None, groups=train_val_groups))
                train_df = train_val_df.iloc[train_idx].copy()
                val_df = train_val_df.iloc[val_idx].copy()

                # Append the data of this fold to a list that will contain all 5 folds
                train_dfs.append(train_df)
                val_dfs.append(val_df)
                test_dfs.append(test_df)

                # Indices for Tino
                train_idx_abs = train_val_idx[train_idx]
                val_idx_abs = train_val_idx[val_idx]
                test_idx_abs = test_idx
                records.append({
                    "config": f"train_val_dbs={train_val_dbs}_test_db={test_db}",
                    "fold": fold_idx,
                    "train_idx": train_idx_abs.tolist(),
                    "val_idx": val_idx_abs.tolist(),
                    "test_idx": test_idx_abs.tolist()
                })
                fold_idx += 1

        else:
            # Split test from train_val because the database for test is different
            train_val_df = train_val_test_df[train_val_test_df['dataset'].isin(train_val_dbs)].copy()
            test_df = train_val_test_df[train_val_test_df['dataset'] == test_db].copy()

            # Perform K-fold on train_val_df
            groups = train_val_df[columns["id"]]
            for train_idx, val_idx in group_kfold.split(train_val_df, y=None, groups=groups):
                train_df = train_val_df.iloc[train_idx].copy()
                val_df = train_val_df.iloc[val_idx].copy()

                # Append the data of this fold to a list that will contain all 5 folds
                train_dfs.append(train_df)
                val_dfs.append(val_df)
                test_dfs.append(test_df)

                # Indices for Tino
                train_idx_abs = train_val_df.index[train_idx].tolist()
                val_idx_abs = train_val_df.index[val_idx].tolist()
                records.append({
                    "config": f"train_val_dbs={train_val_dbs}_test_db={test_db}",
                    "fold": fold_idx,
                    "train_idx": train_idx_abs,
                    "val_idx": val_idx_abs,
                    "test_idx": "all"
                })
                fold_idx += 1

        # Write the indices into a file
        indices_filepath = "results/indices.txt"
        with open(indices_filepath, "a") as f:
            f.write(f"Config train_val/test: {train_val_dbs} / {test_db}\n\n")
            for rec in records:
                f.write(f"Fold {rec['fold']}\n")
                f.write(f"  train_idx ({len(rec['train_idx'])} samples): {rec['train_idx']}\n")
                f.write(f"  val_idx   ({len(rec['val_idx'])} samples): {rec['val_idx']}\n")
                f.write(f"  test_idx  ({len(rec['test_idx'])} samples): {rec['test_idx']}\n\n")

        return train_dfs, val_dfs, test_dfs


    @staticmethod
    def _train_model(train_df, val_df, test_df, fold_dir, fold, columns, nn):
        """
        Prepare data and train a model for a specific fold.

        Args:
            train_df (pandas.DataFrame): Training data
            val_df (pandas.DataFrame): Validation data
            test_df (pandas.DataFrame): Test data
            fold_dir (str): Directory path for saving fold results
            fold (int): Fold number (0-indexed)
            columns (dict): Column mapping configuration
            nn (str): Neural network architecture name
        """

        # Process train, validation, and test DataFrames
        train_fingerprints, train_y, train_ids, train_adducts = train_df[columns['features']], train_df[columns['target']], train_df[columns['id']], train_df[columns['adduct']]
        val_fingerprints, val_y, val_ids, val_adducts = val_df[columns['features']], val_df[columns['target']], val_df[columns['id']], val_df[columns['adduct']]
        test_fingerprints, test_y, test_ids, test_adducts = test_df[columns['features']], test_df[columns['target']], test_df[columns['id']], test_df[columns['adduct']]

        if not columns['features'][0] == "V1":
            X_scaler = MinMaxScaler()
            train_fingerprints = X_scaler.fit_transform(train_fingerprints)
            val_fingerprints = X_scaler.transform(val_fingerprints)
            test_fingerprints = X_scaler.transform(test_fingerprints)
        else:
            X_scaler = None

        # Initialize and fit RobustScaler on training y
        y_scaler = RobustScaler()
        train_y_scaled = y_scaler.fit_transform(train_y.to_numpy().reshape(-1, 1)).flatten()
        val_y_scaled = y_scaler.transform(val_y.to_numpy().reshape(-1, 1)).flatten()
        test_y_scaled = y_scaler.transform(test_y.to_numpy().reshape(-1, 1)).flatten()

        # Encode adducts using the provided encoder
        adduct_encoder = OneHotEncoder()
        adducts_train_encoded = adduct_encoder.fit_transform(train_adducts.to_numpy().reshape(-1, 1)).toarray()
        adducts_val_encoded = adduct_encoder.transform(val_adducts.to_numpy().reshape(-1, 1)).toarray()
        adducts_test_encoded = adduct_encoder.transform(test_adducts.to_numpy().reshape(-1, 1)).toarray()

        # Call to _train_single_fold
        fold_dir = fold_dir + f"_fold{fold + 1}"
        train_a_fold(
            train_data=(train_fingerprints, adducts_train_encoded, train_y_scaled),
            val_data=(val_fingerprints, adducts_val_encoded, val_y_scaled),
            test_data=(test_fingerprints, adducts_test_encoded, test_y_scaled),
            adduct_encoder=adduct_encoder,
            y_scaler=y_scaler,
            fold=fold,
            fold_dir=fold_dir,
            nn=nn
        )

    @staticmethod
    def _write_final_metrics(partial_path):
        """
        Calculate and append final aggregated metrics to results CSV.

        Args:
            partial_path (str): Path prefix for the results CSV file

        Calculates mean ± standard deviation across all folds and appends
        a summary row to the CSV file.
        """

        csv_path = partial_path + "_results.csv"
        print("  - Final evaluation (calculate: mean + std)")

        # Read all data from CSV
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            _ = next(reader)  # Read header row
            values = list(reader)  # Read all remaining rows

        # Convert to numpy array, excluding Fold column
        data_array = np.array(values, dtype=float)[:, 1:]  # Skip Fold column

        # Calculate mean and standard deviation
        means = np.mean(data_array, axis=0)
        stds = np.std(data_array, axis=0, ddof=1)  # ddof=1 uses n-1 in the denominator instead of n

        # Format results as "mean ± std"
        results = [f"{mean:.4f}±{std:.4f}" for mean, std in zip(means, stds)]

        # Append results to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Total"] + results)
