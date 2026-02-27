import base64
import os
import requests
import zipfile
import time

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, inchi
from os import error

import pandas as pd


class ExtractFiles:
    """
    Ensures required dataset files are available in the resources folder.

    If the expected files are not already present, it extracts all zip
    archives found in the data folder directly into the resources folder
    without preserving internal directory structures.
    """

    def __init__(self):
        """
        Initialize paths and trigger extraction if required files
        are not already available in the resources directory.
        """

        # Folders
        resources_folder = "resources"
        data_folder = "data"

        # Expected files
        expected_files = [
            "allccs.csv",
            "ccsbase.csv",
            "metlinccs1.csv",
            "metlinccs2.csv",
            "mobility.csv",
            "smrt.csv",
            "all_datasets_classification.csv",
        ]

        # Return if all expected files exist
        all_files_exist = all(os.path.exists(os.path.join(resources_folder, f))for f in expected_files)
        if all_files_exist: return

        # Extract the files
        self.extract_file(data_folder, resources_folder)

    @staticmethod
    def extract_file(data_folder, resources_folder):
        """
        Extract each zip file in data_folder directly into resources_folder.
        Assumes every zip contains exactly one file and no directory structure.
        """

        os.makedirs(resources_folder, exist_ok=True)

        for file_name in os.listdir(data_folder):
            if file_name.lower().endswith(".zip"):
                zip_path = os.path.join(data_folder, file_name)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(resources_folder)

class UnifyFormats:
    """
    Unified processor for multiple CCS and mobility datasets.

    Standardizes different dataset formats (AllCCS, CCSBase, METLIN-CCS, HMDB, SMRT)
    into a consistent schema with InChI identifiers, adduct filtering, and CCS/mobility values.
    """

    def __init__(self):
        """
        Initialize dataset unifier and execute processing pipeline if output doesn't exist.

        Checks for existing unified dataset at 'resources/all_datasets.csv'. If missing,
        processes all individual datasets with allowed adduct filtering and merges them.

        Allowed adducts: [M+Na]+, [2M+Na]+, [M+H]+, [2M+H]+, [M-H]-, [2M-H]-
        """

        # List of allowed adducts
        self.allowed_adducts = {'[M+Na]+', '[2M+Na]+', '[M+H]+', '[2M+H]+', '[M-H]-', '[2M-H]-'}

        # Return if done
        csv_path = "resources/all_datasets.csv"
        if os.path.exists(csv_path): return
        # Create the dataframes
        allccs_df = self.process_allccs_dataset()
        ccsbase_df = self.process_ccsbase_dataset()
        metlinccs1_df = self.process_metlinccs1_dataset()
        metlinccs2_df = self.process_metlinccs2_dataset()
        mobility_df = self.process_mobility_dataset()
        smrt_df = self.process_smrt_dataset()

        datasets = [allccs_df, ccsbase_df, metlinccs1_df, metlinccs2_df, mobility_df, smrt_df]
        self.merge_and_save(datasets, csv_path)

    def process_allccs_dataset(self, input_file='resources/allccs.csv'):
        """
        Process AllCCS dataset with InChI identifiers.

        Args:
            input_file (str): Path to AllCCS CSV file

        Returns:
            pandas.DataFrame: Standardized DataFrame with columns:
                - dataset: 'allccs'
                - inchi: InChI string
                - smile: SMILES string
                - adduct: Filtered adduct type
                - m/z: Mass-to-charge ratio
                - ccs: Collision cross section value
        """

        # Load the allccs.csv file
        input_df = pd.read_csv(input_file)

        # Filter rows where 'Adduct' is in the allowed list
        filtered_df = input_df[input_df['Adduct'].isin(self.allowed_adducts)].copy()

        # Create new DataFrame with required columns
        new_df = pd.DataFrame({
            'dataset': 'allccs',
            'inchi': filtered_df['InChI'],
            'smile': filtered_df['Structure'],
            'adduct': filtered_df['Adduct'],
            'm/z': filtered_df['m/z'],
            'ccs': filtered_df['CCS']
        })

        # Remove rows where InChI is missing
        new_df = new_df.dropna(subset=['inchi'])

        print(new_df.head())

        return new_df

    def process_ccsbase_dataset(self, input_file='resources/ccsbase.csv'):
        """
        Process CCSBase dataset by converting SMILES to InChI identifiers.

        Args:
            input_file (str): Path to CCSBase CSV file

        Returns:
            pandas.DataFrame: Standardized DataFrame with InChI-converted identifiers.
                            Rows with failed InChI conversion are removed.
        """

        # Load the ccsbase.csv file
        df = pd.read_csv(input_file)

        # Filter rows where 'adduct' is in the allowed list
        filtered_df = df[df['adduct'].isin(self.allowed_adducts)].copy()

        # Function to convert SMILES to InChI
        def smiles_to_inchi(smiles):
            try:
                if pd.isna(smiles) or not isinstance(smiles, str):
                    return None
                else:
                    mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None
                else:
                    inchi_str = inchi.MolToInchi(mol)
                if inchi_str:
                    return inchi_str
                else:
                    return None
            except error as E:
                return None

        # Create new DataFrame with required columns
        new_df = pd.DataFrame({
            'dataset': 'ccsbase',
            'inchi': filtered_df['smi'].apply(smiles_to_inchi),
            'smile': filtered_df['smi'],
            'adduct': filtered_df['adduct'],
            'm/z': filtered_df['m/z'],
            'ccs': filtered_df['ccs']
        })

        # Remove rows where InChI conversion failed
        new_df = new_df.dropna(subset=['inchi'])

        # Reset index
        new_df = new_df.reset_index(drop=True)

        print(new_df.head())
        return new_df

    @staticmethod
    def process_hmdb_dataset(input_file='resources/hmdb.csv'):
        """
        Process HMDB metabolite database without CCS or adduct information.

        Args:
            input_file (str): Path to HMDB CSV file

        Returns:
            pandas.DataFrame: DataFrame with InChI identifiers only.
                            CCS, m/z, and adduct columns are set to None.
        """

        # Load the hmdb.csv file, skipping bad lines
        df = pd.read_csv(input_file, on_bad_lines='skip')

        # Create new DataFrame with required columns
        new_df = pd.DataFrame({
            'dataset': 'hmdb',
            'inchi': df['inchi'],
            'smile': None,
            'adduct': None,
            'm/z': None,
            'ccs': None
        })

        # Remove rows where InChI is missing
        new_df = new_df.dropna(subset=['inchi'])

        # Reset index
        new_df = new_df.reset_index(drop=True)

        print(new_df.head())
        return new_df

    @staticmethod
    def process_metlinccs1_dataset(input_file='resources/metlinccs1.csv'):
        """
        Process METLIN-CCS dataset with multiple CCS measurements and dimer corrections.

        Args:
            input_file (str): Path to METLIN-CCS1 CSV file

        Returns:
            pandas.DataFrame: Standardized DataFrame with:
                - Average CCS from 3 measurements
                - Corrected m/z for dimer adducts
                - Processed adduct notation with charge signs
        """

        # Load the metlinccs1.tsv file
        df = pd.read_csv(input_file)

        # Calculate average CCS from CCS1, CCS2, CCS3
        df['ccs'] = df[['CCS1', 'CCS2', 'CCS3']].mean(axis=1).round(3)

        # Process adduct column and correct m/z for dimers
        def process_adduct_and_mz(row):
            adduct = row['Adduct']
            mz = row['m/z']
            # Add + or - based on the sign in the adduct
            if '+' in adduct:
                charge = '+'
            elif '-' in adduct:
                charge = '-'
            adduct = adduct + charge

            # If dimer, adjust adduct and m/z
            if row['Dimer.1'] == 'Dimer':
                adduct = adduct.replace('M', '2M')
                if charge == '+':
                    if 'Na' in adduct:
                        mz = 2 * mz - 22.989769  # Atomic mass of Na
                    else:  # Assume H
                        mz = 2 * mz - 1.007825  # Atomic mass of H
                else:  # charge == '-'
                    if 'Na' in adduct:
                        mz = 2 * mz + 22.989769  # Atomic mass of Na
                    else:  # Assume H
                        mz = 2 * mz + 1.007825  # Atomic mass of H

            return pd.Series([adduct, mz])

        # Apply adduct and m/z processing
        df[['processed_adduct', 'corrected_mz']] = df.apply(process_adduct_and_mz, axis=1)

        # Create new DataFrame with required columns
        new_df = pd.DataFrame({
            'dataset': 'metlinccs1',
            'inchi': df['inchi'],
            'smile': df['smiles'],
            'adduct': df['processed_adduct'],
            'm/z': df['corrected_mz'],
            'ccs': df['ccs']
        })

        # Remove rows where InChI is missing
        new_df = new_df.dropna(subset=['inchi'])

        # Reset index
        new_df = new_df.reset_index(drop=True)

        print(new_df.head())
        return new_df

    def process_metlinccs2_dataset(self, input_file='resources/metlinccs2.csv'):
        """
        Process METLIN-CCS2 dataset with adduct-specific CCS columns.

        Args:
            input_file (str): Path to METLIN-CCS2 CSV file

        Returns:
            pandas.DataFrame: DataFrame expanded with one row per valid adduct-CCS pair.
                            Each allowed adduct creates a separate row if CCS value exists.
        """

        # Load the metlinccs2.csv file, skipping bad lines
        df = pd.read_csv(input_file, on_bad_lines='skip')

        # Initialize list to store rows for new DataFrame
        rows = []

        # Process each compound for adduct-specific CCS columns
        for _, row in df.iterrows():
            inchi_string = row['InChI']
            if pd.isna(inchi_string) or not isinstance(inchi_string, str):
                continue  # Skip rows with invalid InChI

            for adduct in self.allowed_adducts:
                ccs_col = f'CCS {adduct}'
                if ccs_col in df.columns and pd.notna(row[ccs_col]):
                    try:
                        ccs = float(str(row[ccs_col]).replace(',', '.'))
                        rows.append({
                            'dataset': 'metlinccs2',
                            'inchi': inchi_string,
                            'smile': None,
                            'adduct': adduct,
                            'm/z': None,  # m/z not provided in metlinccs2
                            'ccs': ccs
                        })
                    except (ValueError, TypeError):
                        continue  # Skip invalid CCS values

        # Create new DataFrame from processed rows
        new_df = pd.DataFrame(rows)

        # Remove rows where InChI is missing
        new_df = new_df.dropna(subset=['inchi'])

        # Reset index
        new_df = new_df.reset_index(drop=True)

        print(new_df.head())
        return new_df

    @staticmethod
    def process_mobility_dataset(input_file='resources/mobility.csv'):
        """
        Process raw mobility dataset from CSV file into a structured DataFrame.

        The function reads a CSV file without headers from 'resources/mobility.csv' and
        transforms it into a formatted DataFrame with standardized column names for
        mobility spectrometry data.

        Returns:
            pandas.DataFrame: A DataFrame with renamed columns
        """

        # Load raw CSV (no header)
        df = pd.read_csv(input_file, header=None)

        new_df = pd.DataFrame({
            'dataset': df[0],
            'inchi': df[1],
            'smile': df[2],
            'adduct': df[3],
            'm/z': df[4],
            'ccs': df[5]
        })

        return new_df

    @staticmethod
    def process_smrt_dataset(input_file='resources/smrt.csv'):
        """
        Process SMRT mobility dataset using retention time as target value.

        Args:
            input_file (str): Path to SMRT CSV file

        Returns:
            pandas.DataFrame: Standardized DataFrame with:
                - inchi: InChI identifiers
                - ccs: Retention time values (used as prediction target)
                - adduct, m/z, smile: Set to None (not provided in SMRT)
        """

        # Load the smrt.csv file, skipping bad lines
        df = pd.read_csv(input_file, on_bad_lines='skip')

        # Create new DataFrame with required columns
        new_df = pd.DataFrame({
            'dataset': 'smrt',
            'inchi': df['inchi'],
            'smile': None,
            'adduct': None,
            'm/z': None,
            'ccs': df['rt']
        })

        # Remove rows where InChI is missing
        new_df = new_df.dropna(subset=['inchi'])

        # Reset index
        new_df = new_df.reset_index(drop=True)

        print(new_df.head())
        return new_df

    @staticmethod
    def merge_and_save(datasets, csv_path):
        """
        Merge multiple standardized datasets and save to CSV.

        Args:
            datasets (list): List of pandas DataFrames to concatenate
            csv_path (str): Output path for merged CSV file

        Returns:
            None: Saves merged DataFrame to disk
        """

        # Merge DataFrames by concatenating vertically
        merged_df = pd.concat(datasets, axis=0, ignore_index=True)

        # Save DataFrame to CSV
        merged_df.to_csv(csv_path, index=False)

        print(f"DataFrame saved")



class PrepareDataset:
    """
    Prepares molecular datasets by completing missing SMILES using PubChem and RDKit.

    Handles two-stage SMILES completion: first attempts PubChem API lookup for canonical
    SMILES, then uses RDKit for 3D structure generation and SMILES conversion when needed.
    """

    def __init__(self):
        """
        Initialize dataset preparer and execute SMILES completion pipeline if needed.

        Checks for existing completed dataset at 'resources/all_datasets_completed.csv'.
        If missing, performs sequential SMILES completion using PubChem API followed
        by RDKit 3D structure generation.
        """

        # Return if done
        csv_path_to_completed = "resources/all_datasets_completed.csv"
        if os.path.exists(csv_path_to_completed): return

        self.fetch_smiles_from_pubchem()
        self.complete_smiles_with_rdkit(csv_path_to_completed)

    @staticmethod
    def fetch_smiles_from_pubchem():
        """
        Fetch missing SMILES from PubChem using InChI-to-CID-to-SMILES conversion.

        Queries PubChem REST API with InChI strings to obtain canonical SMILES.
        Implements rate limiting (0.2s delay) and incremental saving every 100 updates.
        Saves intermediate results to 'resources/all_datasets_pubchem_smiles.csv'.

        Note: HMDB entries are excluded as they lack structural information.
        """

        # Paths
        original_csv = "resources/all_datasets.csv"
        output_csv = "resources/all_datasets_pubchem_smiles.csv"

        # Return if done
        if os.path.exists(output_csv): return

        # Load the CSV into a DataFrame
        df = pd.read_csv(original_csv)

        # Function to get SMILES from InChI using PubChem (InChI -> CID -> SMILES, with fallback)
        def get_smiles_from_inchi(inchi):
            """
            Retrieve canonical SMILES from PubChem using InChI identifier.

            Implements two-step PubChem API query:
            1. Convert InChI to PubChem CID using compound/cids endpoint
            2. Fetch SMILES via property endpoint with fallback to full record search

            Args:
                inchi (str): InChI string for chemical compound

            Returns:
                str or None: Canonical SMILES string if found, None if retrieval fails
            """

            try:
                # Step 1: Get CID from InChI
                cid_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchi/cids/JSON"
                cid_res = requests.post(cid_url, data={"inchi": inchi}, timeout=30)
                cid_res.raise_for_status()
                cid_data = cid_res.json()
                cids = cid_data.get("IdentifierList", {}).get("CID", [])
                if not cids:
                    print(f"No CID found for InChI: {inchi}")
                    return None
                cid = cids[0]

                # Step 2a: Try property endpoint
                prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
                prop_res = requests.get(prop_url, timeout=30)
                if prop_res.ok:
                    prop_data = prop_res.json()
                    props = prop_data.get("PropertyTable", {}).get("Properties", [])
                    if props and "CanonicalSMILES" in props[0]:
                        return props[0]["CanonicalSMILES"]

                # Step 2b: Fallback — fetch full record JSON
                record_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
                record_res = requests.get(record_url, timeout=30)
                record_res.raise_for_status()
                record_data = record_res.json()

                # Recursively search for "SMILES"
                def find_smiles(sections):
                    for sec in sections:
                        toc = sec.get("TOCHeading", "")
                        if "SMILES" in toc:
                            for info in sec.get("Information", []):
                                val = info.get("Value", {}).get("StringWithMarkup", [])
                                if val:
                                    return val[0]["String"]
                        sub = find_smiles(sec.get("Section", []))
                        if sub:
                            return sub
                    return None

                sections = record_data.get("Record", {}).get("Section", [])
                return find_smiles(sections)

            except Exception as e:
                print(f"Error fetching SMILES for InChI {inchi}: {e}")
                return None

        a = 0
        # Iterate through the DataFrame row by row
        for index, row in df.iterrows():
            if pd.isna(row['smile']) or row['smile'] is None:
                inchi = row['inchi']
                if pd.notna(inchi) and isinstance(inchi, str):
                    print(f"Fetching SMILES for InChI at index {index}: {inchi}")
                    smile = get_smiles_from_inchi(inchi)
                    if smile:
                        df.at[index, 'smile'] = smile
                        print(f"Updated SMILES at index {index}: {smile}")
                        a += 1
                        if a % 100 == 0:
                            df.to_csv(output_csv, index=False)
                            print(f"Saved updated CSV to {output_csv} after {a} updates")
                    else:
                        print(f"Failed to fetch SMILES for InChI at index {index}")
                    time.sleep(0.2)  # respect rate limit of 0.2 if necessary
                else:
                    print(f"Invalid or missing InChI at index {index}: {inchi}")

        # Save final DataFrame
        df.to_csv(output_csv, index=False)
        print(f"Completed processing. Final CSV saved at {output_csv}")

    @staticmethod
    def complete_smiles_with_rdkit(csv_path_to_completed):
        """
        Complete remaining missing SMILES using RDKit 3D structure generation.

        Processes molecules with missing SMILES by:
        1. Converting InChI to RDKit molecule
        2. Generating and optimizing 3D coordinates (MMFF force field)
        3. Converting back to SMILES representation
        Removes HMDB entries and any rows where SMILES cannot be generated.

        Args:
            csv_path_to_completed (str): Output path for final completed dataset
        """

        # Define file path
        csv_path = "resources/all_datasets_pubchem_smiles.csv"
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)
        print("Initial len", len(df))
        # Remove rows where 'dataset' is 'hmdb' (these smiles can't be properly generated in 3D, neither with pubchem nor with rdkit)
        df = df[df['dataset'] != 'hmdb']
        # Remove rows where 'inchi' is None or NaN
        df = df.dropna(subset=['inchi'], how='all')
        # Check for None or NaN in the 'smile' column
        for idx, row in df.iterrows():
            if pd.isna(row['smile']) or row['smile'] is None:
                # Get the InChI string
                inchi = row['inchi']
                if pd.isna(inchi) or inchi is None:
                    print(f"Skipping row {idx}: Both SMILES and InChI are None or NaN")
                    continue
                try:
                    # Convert InChI to RDKit molecule
                    mol = Chem.MolFromInchi(inchi)
                    if mol is None:
                        print(f"Failed to create molecule from InChI at row {idx}")
                        continue
                    # Generate 3D coordinates
                    AllChem.EmbedMolecule(mol, randomSeed=42) # Fixed seed for reproducibility
                    AllChem.MMFFOptimizeMolecule(mol) # Optimize 3D structure
                    AllChem.MMFFOptimizeMolecule(mol) # Optimize 3D structure
                    # Convert to SMILES
                    smile = Chem.MolToSmiles(mol)
                    # Update the DataFrame
                    df.at[idx, 'smile'] = smile
                    print(f"Updated SMILE at row {idx}: {smile}")
                except Exception as e:
                    print(f"Error processing row {idx}: {str(e)}")
                    continue
        # Remove rows where 'smile' is None or NaN
        df = df.dropna(subset=['smile'], how='all')
        # Save the updated DataFrame back to the CSV
        df.to_csv(csv_path_to_completed, index=False)
        print(f"Updated CSV saved to {csv_path_to_completed}")
        print("Final len", len(df))


class UnifyEmbeddings:
    """
    Unifies molecular embeddings by finding common compounds across different representation files.

    Identifies and retains only compounds present in all molecular representation formats
    (UniMol, ChemBERTa, MolFormer, MolMIM, fingerprints) to ensure consistent dataset coverage.
    """

    def __init__(self):
        """
        Initialize embedding unifier and execute unification if needed.

        Checks for existing unified embedding files. If any are missing, triggers
        the unification process to filter datasets to their common intersection.

        Expected unified files:
        - all_datasets_{representation}_unified.csv for each embedding type
        """

        folder_path = Path("resources")
        file_names = [
            "all_datasets_unimol.csv",
            "all_datasets_chemberta.csv",
            "all_datasets_molformer.csv",
            "all_datasets_molmim.csv",
            "all_datasets_fingerprints.csv"
        ]
        unified_names = [name.replace(".csv", "_unified.csv") for name in file_names]
        unified_paths = [folder_path / name for name in unified_names]

        if all(path.exists() for path in unified_paths): return

        self.unify()

    @staticmethod
    def unify():
        """
        Unify embeddings by intersecting compounds across all representation datasets.

        Loads embedding datasets, identifies common compounds based on structural
        identifiers (dataset, inchi, smile, adduct, m/z, ccs), and saves filtered
        versions containing only the intersection.

        Process:
        1. Load available embedding CSV files
        2. Create signature sets for each dataset
        3. Compute set intersection of all signatures
        4. Filter each dataset to common compounds
        5. Save unified datasets with '_unified' suffix
        """

        folder_path = Path("resources")
        required_columns = ['dataset', 'inchi', 'smile', 'adduct', 'm/z', 'ccs']

        file_names = [
            "all_datasets_unimol.csv",
            "all_datasets_chemberta.csv",
            "all_datasets_molformer.csv",
            "all_datasets_molmim.csv",
            "all_datasets_fingerprints.csv"
        ]
        file_paths = [folder_path / name for name in file_names]

        print("Loading files...")
        dataframes = []
        loaded_paths = []

        for file_path in file_paths:
            if file_path.exists():
                print(f"  Reading {file_path.name}")
                df = pd.read_csv(file_path)
                dataframes.append(df)
                loaded_paths.append(file_path)
            else:
                print(f"  Skipping {file_path.name} (file not found)")

        print(f"Loaded {len(dataframes)} files.\n")

        if len(dataframes) == 0:
            print("No files loaded. Exiting.")
            return

        print("Building signature sets...")
        sets = []
        for i, df in enumerate(dataframes):
            print(f"  Processing dataset {i+1}")
            subset = df[required_columns].astype(str)
            sets.append({tuple(row) for row in subset.to_numpy()})

        print("\nFinding common rows...")
        common_rows = set.intersection(*sets)
        print(f"Common row patterns across all datasets: {len(common_rows)}\n")

        print("Filtering datasets...")
        filtered_dfs = []
        for i, df in enumerate(dataframes):
            print(f"  Filtering dataset {i+1}")
            sig = df[required_columns].astype(str).apply(tuple, axis=1)
            mask = sig.isin(common_rows)

            kept = mask.sum()
            removed = len(df) - kept
            print(f"    Kept: {kept} | Removed: {removed}")

            filtered_dfs.append(df[mask].copy())

        print("\nSaving results...")
        for filtered_df, original_path in zip(filtered_dfs, loaded_paths):
            new_filepath = original_path.with_name(original_path.stem + "_unified.csv")
            filtered_df.to_csv(new_filepath, index=False)
            print(f"  Saved {new_filepath.name}")

        print("\nDone.")
