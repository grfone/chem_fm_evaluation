import torch
import os
import requests
import glob
import docker
import time
import subprocess

import pandas as pd
import numpy as np

from unimol_tools import UniMolRepr
from transformers import AutoTokenizer, AutoModel


class GenerateEmbeddings:
    """
    A unified class to generate various molecular embeddings and fingerprints.
    Upon initialization, it sequentially generates UniMol, ChemBERTa, MoLFormer,
    MolMIM embeddings, and copies Alvadesc fingerprints by calling respective methods.
    """

    def __init__(self):
        self.generate_unimol_embeddings()
        self.generate_chemberta_embeddings()
        self.generate_molformer_embeddings()
        self.generate_molmim_embeddings()
        self.copy_alvadesc_fingerprints()

    def generate_unimol_embeddings(self):
        """
        Generate UniMol embeddings using Uni-Mol v2 (1.1B).
        UniMol will automatically download weights/dictionaries into its default cache.
        Reads from 'resources/all_datasets_completed.csv' and saves to 'resources/all_datasets_unimol.csv'.
        """
        # Full filepath to save the unimol embeddings
        unimol_csv = "resources/all_datasets_unimol.csv"

        # Check if the file was already generated
        if os.path.exists(unimol_csv):
            return

        # Initialize UniMolRepr with Uni-Mol v2 1.1B settings
        model = UniMolRepr(
            data_type="molecule",
            remove_hs=False,  # Keep hydrogens (important for CCS)
            model_name="unimolv2",  # Use Uni-Mol v2
            model_size="1.1B",  # 1.1B parameter model
            dict_name=None  # UniMol finds mol_dict.txt automatically
        )

        # Define file path
        csv_path_to_completed = "resources/all_datasets_completed.csv"

        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path_to_completed)

        # Check if fingerprint columns already exist, create only if they don't
        fingerprint_columns = [f"unimol_{i}" for i in range(1536)]
        existing_columns = set(df.columns)
        missing_columns = [col for col in fingerprint_columns if col not in existing_columns]

        if missing_columns:
            for col in missing_columns:
                df[col] = np.nan  # Initialize missing columns with NaN

        # Iterate through rows to compute fingerprints
        for idx, row in df.iterrows():
            # Skip rows where fingerprint columns are already populated
            if not pd.isna(row[fingerprint_columns]).any():
                continue

            smile = row['smile']
            try:
                # Compute fingerprint using the provided function
                fingerprint = self._get_unimol_fingerprint(model, smile)
                # Fill the fingerprint columns for this row
                df.loc[idx, fingerprint_columns] = fingerprint
                print(f"Computed fingerprint for row {idx}")
                print("Tamaño del dataset:", len(df))
            except Exception as e:
                print(f"Error computing fingerprint for row {idx}: {e}")
                df = df.drop(idx)  # Remove row if fingerprint computation fails

            # Save dataset every 10000 rows
            if idx % 10000 == 0 and idx != 0:
                df.to_csv(f'resources/dataset_checkpoint_{idx}.csv', index=False)
                print(f"Saved dataset at row {idx}")

        # Save the updated DataFrame back to the CSV
        df.to_csv(unimol_csv, index=False)
        print(f"Updated CSV with fingerprints saved to {unimol_csv}")

        # Remove checkpoints after it finishes
        checkpoint_files = glob.glob('resources/dataset_checkpoint_*.csv')
        for checkpoint_file in checkpoint_files:
            os.remove(checkpoint_file)

    @staticmethod
    def _get_unimol_fingerprint(model, smile):
        """
        Generate a molecular fingerprint for a given SMILES string using Uni-Mol v2 1.1B.

        Args:
            model (UniMolRepr): The initialized UniMol model.
            smile (str): Input SMILES string.

        Returns:
            np.ndarray: Molecular fingerprint as a fixed-length vector.
        """
        repr_output = model.get_repr([smile], return_atomic_reprs=False)
        return np.array(repr_output[0])

    def generate_chemberta_embeddings(self):
        """
        Generate ChemBERTa embeddings using the DeepChem/ChemBERTa-100M-MLM model.
        Reads from 'resources/all_datasets_completed.csv' and saves to 'resources/all_datasets_chemberta.csv'.
        """
        # Full filepath to save the chemberta embeddings
        chemberta_csv = "resources/all_datasets_chemberta.csv"

        # Check if the file was already generated
        if os.path.exists(chemberta_csv):
            return

        # Load tokenizer and model (768 features) from https://huggingface.co/DeepChem/models
        model_name = "DeepChem/ChemBERTa-100M-MLM"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Define file path
        csv_path = "resources/all_datasets_completed.csv"

        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)

        # Check if embedding columns already exist, create only if they don't
        embedding_columns = [f"chemberta_{i}" for i in range(768)]
        existing_columns = set(df.columns)
        missing_columns = [col for col in embedding_columns if col not in existing_columns]

        if missing_columns:
            for col in missing_columns:
                df[col] = np.nan  # Initialize missing columns with NaN

        # Find rows that need computation (any NaN in embedding columns)
        rows_to_compute = df[df[embedding_columns].isna().any(axis=1)].index.tolist()
        if not rows_to_compute:
            print("All embeddings already computed.")
            return

        # Get SMILES for rows that need computation
        smiles_to_compute = df.loc[rows_to_compute, 'smile'].tolist()

        # Compute embeddings in batches
        batch_size = 32  # Adjust based on GPU memory
        embeddings = []
        failed_indices = []

        for i in range(0, len(smiles_to_compute), batch_size):
            batch_smiles = smiles_to_compute[i:i + batch_size]
            batch_indices = rows_to_compute[i:i + batch_size]

            try:
                batch_embeddings = self._get_chemberta_batch_embeddings(tokenizer, model, device, batch_smiles)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error computing batch starting at index {i}: {e}")
                for j, smile in enumerate(batch_smiles):
                    try:
                        # Fallback to single computation
                        single_embedding = self._get_chemberta_batch_embeddings(tokenizer, model, device, [smile])
                        embeddings.extend(single_embedding)
                    except Exception as single_e:
                        print(f"Error computing embedding for SMILE '{smile}' at row {batch_indices[j]}: {single_e}")
                        failed_indices.append(batch_indices[j])

        # Fill the embeddings
        if embeddings:
            embeddings = np.array(embeddings)
            fill_indices = [idx for idx in rows_to_compute if idx not in failed_indices]

            emb_df = pd.DataFrame(
                embeddings[:len(fill_indices)],
                index=fill_indices,
                columns=embedding_columns
            )
            df.loc[fill_indices, embedding_columns] = emb_df

        # Drop failed rows
        if failed_indices:
            df = df.drop(failed_indices)
            print(f"Dropped {len(failed_indices)} rows due to computation errors.")

        # Save the updated DataFrame back to the CSV
        df.to_csv(chemberta_csv, index=False)
        print(f"Updated CSV with embeddings saved to {chemberta_csv}")

    @staticmethod
    def _get_chemberta_batch_embeddings(tokenizer, model, device, smiles_list):
        """
        Generate molecular embeddings for a batch of SMILES strings using ChemBERTa.

        Args:
            tokenizer (AutoTokenizer): The tokenizer for ChemBERTa.
            model (AutoModel): The ChemBERTa model.
            device (torch.device): The device to run the model on.
            smiles_list (list): List of SMILES strings.

        Returns:
            np.ndarray: Array of embeddings with shape (batch_size, 768).
        """
        inputs = tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, 768)

        pool_emb = hidden_states[:, 0, :]  # (batch, 768)

        return pool_emb.cpu().numpy()  # directly as ndarray

    def generate_molformer_embeddings(self):
        """
        Generate MoLFormer embeddings using the ibm/MoLFormer-XL-both-10pct model.
        Reads from 'resources/all_datasets_completed.csv' and saves to 'resources/all_datasets_molformer.csv'.
        """
        # Full filepath to save the molformer embeddings
        molformer_csv = "resources/all_datasets_molformer.csv"

        # Check if the file was already generated
        if os.path.exists(molformer_csv):
            return

        # Load tokenizer and model from https://huggingface.co/ibm/MoLFormer-XL-both-10pct
        model_name = "ibm/MoLFormer-XL-both-10pct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, deterministic_eval=True, trust_remote_code=True)

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Define file path
        csv_path = "resources/all_datasets_completed.csv"

        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)

        # MoLFormer-XL output dimension is 768 (same as ChemBERTa for consistency)
        embedding_columns = [f"molformer_{i}" for i in range(768)]
        existing_columns = set(df.columns)
        missing_columns = [col for col in embedding_columns if col not in existing_columns]

        if missing_columns:
            for col in missing_columns:
                df[col] = np.nan  # Initialize missing columns with NaN

        # Find rows that need computation (any NaN in embedding columns)
        rows_to_compute = df[df[embedding_columns].isna().any(axis=1)].index.tolist()
        if not rows_to_compute:
            print("All embeddings already computed.")
            return

        # Get SMILES for rows that need computation
        smiles_to_compute = df.loc[rows_to_compute, 'smile'].tolist()

        # Compute embeddings in batches
        batch_size = 32  # Adjust based on GPU memory
        embeddings = []
        failed_indices = []

        for i in range(0, len(smiles_to_compute), batch_size):
            batch_smiles = smiles_to_compute[i:i + batch_size]
            batch_indices = rows_to_compute[i:i + batch_size]

            try:
                batch_embeddings = self._get_molformer_batch_embeddings(tokenizer, model, device, batch_smiles)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error computing batch starting at index {i}: {e}")
                for j, smile in enumerate(batch_smiles):
                    try:
                        # Fallback to single computation
                        single_embedding = self._get_molformer_batch_embeddings(tokenizer, model, device, [smile])
                        embeddings.extend(single_embedding)
                    except Exception as single_e:
                        print(f"Error computing embedding for SMILE '{smile}' at row {batch_indices[j]}: {single_e}")
                        failed_indices.append(batch_indices[j])

        # Fill the embeddings
        if embeddings:
            embeddings = np.array(embeddings)
            fill_indices = [idx for idx in rows_to_compute if idx not in failed_indices]

            emb_df = pd.DataFrame(
                embeddings[:len(fill_indices)],
                index=fill_indices,
                columns=embedding_columns
            )
            df.loc[fill_indices, embedding_columns] = emb_df

        # Drop failed rows
        if failed_indices:
            df = df.drop(failed_indices)
            print(f"Dropped {len(failed_indices)} rows due to computation errors.")

        # Save the updated DataFrame back to the CSV
        df.to_csv(molformer_csv, index=False)
        print(f"Updated CSV with embeddings saved to {molformer_csv}")

    @staticmethod
    def _get_molformer_batch_embeddings(tokenizer, model, device, smiles_list):
        """
        Generate molecular embeddings for a batch of SMILES strings using MoLFormer-XL.

        Args:
            tokenizer (AutoTokenizer): The tokenizer for MoLFormer.
            model (AutoModel): The MoLFormer model.
            device (torch.device): The device to run the model on.
            smiles_list (list): List of SMILES strings.

        Returns:
            np.ndarray: Array of embeddings with shape (batch_size, 768).
        """
        inputs = tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pool_emb = outputs.pooler_output  # (batch, 768)

        return pool_emb.cpu().numpy()  # directly as ndarray

    def generate_molmim_embeddings(self):
        """
        Generate MolMIM embeddings using a programmatically managed container.
        Starts the container, ensures it's ready, processes the data, then stops the container.
        Reads from and modifies 'resources/all_datasets_completed.csv', saves to 'resources/all_datasets_molmim.csv'.
        """
        # Full filepath to save the molmim embeddings
        molmim_csv = "resources/all_datasets_molmim.csv"
        # Check if the file was already generated
        if os.path.exists(molmim_csv):
            return

        api_url = "http://localhost:8000/embedding"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        batch_size = 100
        session = requests.Session()

        self._check_no_smile_exceeds_128_characters()

        # Start the container
        container = self._start_molmim_container()

        try:
            # Wait until the API is ready
            self._wait_for_container_ready(api_url, headers)

            csv_path = "resources/all_datasets_completed.csv"
            df = pd.read_csv(csv_path)

            # Ensure embedding columns exist
            fingerprint_columns = [f"molmim_{i}" for i in range(512)]
            for col in fingerprint_columns:
                if col not in df.columns:
                    df[col] = np.nan

            indices_to_process = df.index[df[fingerprint_columns].isna().any(axis=1)].tolist()
            print(f"Processing {len(indices_to_process)} rows for embeddings")

            for start_idx in range(0, len(indices_to_process), batch_size):
                batch_indices = indices_to_process[start_idx:start_idx + batch_size]
                batch_smiles = df.loc[batch_indices, "smile"].tolist()

                try:
                    embeddings = self._get_molmim_fingerprint(session, api_url, headers, batch_smiles)
                    for idx, emb in zip(batch_indices, embeddings):
                        df.loc[idx, fingerprint_columns] = emb
                        print(f"Computed MolMIM fingerprint for row {idx}")
                except Exception as e:
                    print(f"Error at batch starting {start_idx}: {e}")

                # Save periodic checkpoints
                if start_idx % 10000 == 0 and start_idx != 0:
                    checkpoint = f"resources/dataset_checkpoint_{start_idx}.csv"
                    df.to_csv(checkpoint, index=False)
                    print(f"Checkpoint saved at {checkpoint}")

            # Final save
            df.to_csv(molmim_csv, index=False)
            print(f"Updated CSV with MolMIM embeddings saved to {molmim_csv}")

            for ckpt in glob.glob("resources/dataset_checkpoint_*.csv"):
                os.remove(ckpt)

            self._remove_nans_from_csv(csv_path)

        finally:
            # Stop the container (will be removed due to auto_remove=True)
            if container:
                container.stop()
                print("MolMIM container stopped.")

    @staticmethod
    def _start_molmim_container():
        """
        Start the MolMIM Docker container programmatically.
        Includes hardcoded NGC_API_KEY and automatic setup.
        """

        client = docker.from_env()

        # Hardcoded credentials and paths
        NGC_API_KEY = "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"
        LOCAL_NIM_CACHE = os.path.expanduser("~/.cache/nim")

        # Ensure cache directory exists
        os.makedirs(LOCAL_NIM_CACHE, exist_ok=True)

        # Configure Docker for NVIDIA runtime (if not already configured)
        print("Configuring Docker for NVIDIA runtime...")
        try:
            subprocess.run(['sudo', 'nvidia-ctk', 'runtime', 'configure', '--runtime=docker'],
                           check=False, capture_output=True)
            subprocess.run(['sudo', 'systemctl', 'restart', 'docker'],
                           check=False, capture_output=True)
        except Exception as e:
            print(f"Note: Docker configuration may already be set up: {e}")

        # Log in to NGC Docker registry
        print("Logging in to NGC Docker registry...")
        try:
            login_result = subprocess.run(
                ['docker', 'login', 'nvcr.io', '--username', '$oauthtoken', '--password-stdin'],
                input=NGC_API_KEY.encode(),
                capture_output=True,
                text=True
            )
            if login_result.returncode != 0:
                print(f"Docker login output: {login_result.stdout}")
                print(f"Docker login error: {login_result.stderr}")
            else:
                print("Docker login successful")
        except Exception as e:
            print(f"Note: Docker login may have failed, continuing: {e}")

        # Pull the MolMIM image (optional - will pull if not exists)
        print("Checking for MolMIM image...")
        try:
            client.images.pull('nvcr.io/nim/nvidia/molmim:1.0.0')
            print("MolMIM image pulled successfully")
        except Exception as e:
            print(f"Note: Image may already exist: {e}")

        # Start the container
        print("Starting MolMIM container...")
        try:
            container = client.containers.run(
                'nvcr.io/nim/nvidia/molmim:1.0.0',
                detach=True,
                runtime='nvidia',
                environment={
                    'CUDA_VISIBLE_DEVICES': '0',
                    'NGC_API_KEY': NGC_API_KEY
                },
                shm_size='2g',
                ports={'8000/tcp': 8000},
                volumes={
                    LOCAL_NIM_CACHE: {'bind': '/home/nvs/.cache/nim', 'mode': 'rw'}
                },
                auto_remove=True  # Emulates --rm; container will be removed after stop
            )
            print(f"Container started with ID: {container.id}")
            return container
        except Exception as e:
            print(f"Error starting container: {e}")
            # Try without explicit runtime (Docker might handle it automatically)
            try:
                container = client.containers.run(
                    'nvcr.io/nim/nvidia/molmim:1.0.0',
                    detach=True,
                    environment={
                        'CUDA_VISIBLE_DEVICES': '0',
                        'NGC_API_KEY': NGC_API_KEY
                    },
                    shm_size='2g',
                    ports={'8000/tcp': 8000},
                    volumes={
                        LOCAL_NIM_CACHE: {'bind': '/home/nvs/.cache/nim', 'mode': 'rw'}
                    },
                    auto_remove=True
                )
                print(f"Container started (without explicit runtime) with ID: {container.id}")
                return container
            except Exception as e2:
                raise RuntimeError(f"Failed to start container with both methods: {e2}")

    @staticmethod
    def _wait_for_container_ready(api_url, headers):
        """
        Wait until the MolMIM API is responsive.
        """
        max_retries = 60  # Up to 2 minutes (2s per retry)
        retries = 0
        while retries < max_retries:
            try:
                test_smiles = ["C"]  # Simple methane SMILES for testing
                payload = {"sequences": test_smiles}
                response = requests.post(api_url, headers=headers, json=payload, timeout=5)
                response.raise_for_status()
                print("MolMIM container is ready.")
                return
            except Exception as e:
                print(f"Waiting for container to be ready... (Attempt {retries + 1}/{max_retries}): {e}")
                time.sleep(2)
                retries += 1
        raise TimeoutError("MolMIM container did not become ready in time.")

    @staticmethod
    def _check_no_smile_exceeds_128_characters():
        """
        Remove rows from 'resources/all_datasets_completed.csv' where SMILES exceed 128 characters.
        """
        # Define file path
        csv_path = "resources/all_datasets_completed.csv"
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)
        # Remove rows where the length of the string in the 'smile' column exceeds 128
        df = df[df['smile'].str.len() <= 128]
        # Save the updated DataFrame back to the CSV
        df.to_csv(csv_path, index=False)
        print(f"Removed long smiles that trigger an error within the Docker")
        print("Final len", len(df))

    @staticmethod
    def _get_molmim_fingerprint(self, session, api_url, headers, smiles):
        """Send SMILES batch to the local MolMIM endpoint and return embeddings."""
        payload = {"sequences": smiles}
        response = session.post(api_url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise ValueError("No embeddings returned from MolMIM response.")
        return [np.array(e) for e in embeddings]

    def copy_alvadesc_fingerprints(self):
        """
        Copy Alvadesc fingerprints from individual dataset files into 'resources/all_datasets_completed.csv'.
        """
        # Full filepath to save the molmim embeddings
        fingerprints_csv = "resources/all_datasets_fingerprints.csv"

        # Check if the file was already generated
        if os.path.exists(fingerprints_csv):
            return

        csv_path = "resources/all_datasets_completed.csv"
        df = pd.read_csv(csv_path)

        # Ensure fingerprint columns exist (V1 to V2214)
        fingerprint_columns = [f"V{i}" for i in range(1, 2215)]
        for col in fingerprint_columns:
            if col not in df.columns:
                df[col] = np.nan

        # Define the dataset files and their matching columns
        dataset_files = {
            'allccs': {'file': 'resources/allccs.csv', 'match_col': 'InChI'},
            'ccsbase': {'file': 'resources/ccsbase.csv', 'match_col': 'smi'},
            'metlinccs1': {'file': 'resources/metlinccs1.csv', 'match_col': 'inchi'},
            'metlinccs2': {'file': 'resources/metlinccs2.csv', 'match_col': 'InChI'},
            'mobility': {'file': 'resources/mobility.csv', 'match_col': 'inchi'},
            'smrt': {'file': 'resources/smrt.csv', 'match_col': 'inchi'}
        }

        # Load each source dataset once
        source_dfs = {}
        for dataset, info in dataset_files.items():
            if os.path.exists(info['file']):
                source_dfs[dataset] = pd.read_csv(info['file'])
            else:
                print(f"Warning: File {info['file']} not found. Skipping dataset {dataset}.")

        # Process each row in all_datasets.csv
        for idx, row in df.iterrows():
            dataset = row['dataset']
            if dataset not in dataset_files:
                print(f"Skipping row {idx}: Unknown dataset {dataset}")
                continue

            if dataset not in source_dfs:
                print(f"Skipping row {idx}: Source data for {dataset} not loaded")
                continue

            source_df = source_dfs[dataset]
            match_col = dataset_files[dataset]['match_col']
            match_value = row['inchi'] if 'inchi' in match_col.lower() else row['smile']

            if pd.isna(match_value):
                print(f"Skipping row {idx}: Match value ({match_col}) is NaN")
                continue

            # Find matching row in source
            match_row = source_df[source_df[match_col] == match_value]
            if match_row.empty:
                print(f"No match found for row {idx} in {dataset} using {match_col}={match_value}")
                continue

            # Assume first match if multiple (though should be unique)
            match_row = match_row.iloc[0]

            if dataset == 'smrt':
                # Shift V0 to V2213 to V1 to V2214
                fp_values = [match_row.get(f"V{i}", np.nan) for i in range(0, 2214)]
            else:
                # Standard V1 to V2214
                fp_values = [match_row.get(f"V{i}", np.nan) for i in range(1, 2215)]

            # Update the DataFrame
            df.loc[idx, fingerprint_columns] = fp_values
            print(f"Copied fingerprints for row {idx} from {dataset}")

        # Save the updated DataFrame
        df.to_csv(fingerprints_csv, index=False)
        print(f"Updated CSV with fingerprints saved to {fingerprints_csv}")

        self._remove_nans_from_csv(fingerprints_csv)

    @staticmethod
    def _remove_nans_from_csv(self, csv_path):
        """
        Remove rows from the given CSV that contain 5 or more NaNs.

        Args:
            csv_path (str): Path to the CSV file to clean.
        """
        # Read the file
        df = pd.read_csv(csv_path)

        # Count the number of rows
        rows_init = len(df)

        # Delete row whose embedding was not generated (contain 5 NaNs or more)
        df = df.dropna(thresh=len(df.columns) - 5)

        # Overwrite the original file with the cleaned DataFrame
        df.to_csv(csv_path, index=False)

        # Count the number of deleted rows
        number_of_deleted_rows = rows_init - len(df)

        # Print the number of deleted rows
        print(f"Deleting {number_of_deleted_rows} rows from {csv_path}")

