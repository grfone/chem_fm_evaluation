import pandas


def load_data(csv_filepath, dataset_name, columns):
    """
    Load and filter dataset from unified CSV file.

    Args:
        csv_filepath (str): Path to unified dataset CSV file
        dataset_name (str): Name of dataset to extract ('ccsbase', 'metlinccs', etc.)
        columns (dict): Dictionary specifying columns to keep with keys:
                       'dataset', 'id', 'features', 'adduct', 'target'

    Returns:
        pandas.DataFrame: Filtered DataFrame containing only specified dataset
                         and required columns for training.

    Note:
        - For 'metlinccs', combines 'metlinccs1' and 'metlinccs2' datasets
        - Automatically flattens feature columns when provided as list
    """

    print(f"  - Loading data for dataset '{dataset_name}'")

    # Load the CSV file without specifying columns to get all
    df = pandas.read_csv(csv_filepath)

    # Filter rows based on dataset_name
    if dataset_name == "metlinccs":
        df = df[df['dataset'].isin(['metlinccs1', 'metlinccs2'])]
        df['dataset'] = 'metlinccs'
    else:
        df = df[df['dataset'] == dataset_name]

    # Keep only the columns specified in the columns dictionary
    keep_columns = []
    for value in columns.values():
        if isinstance(value, list):
            keep_columns.extend(value)
        else:
            keep_columns.append(value)
    df = df[[col for col in keep_columns if col in df.columns]]

    print(f"    ◦ Loaded {len(df)} rows for dataset '{dataset_name}'")

    return df


def handle_duplicates(train_val_test_df, columns, ccs_threshold, train_val_dbs):
    """Check and handle duplicates based on inchi and adduct, returning deduplicated DataFrame.

    Args:
        train_val_test_df (pd.DataFrame): Combined DataFrame containing training, validation, and test data.
        columns (dict): columns in the dataset
        ccs_threshold (float): Maximum allowable relative difference in CCS values for duplicates.
        train_val_dbs (list): List of dataset names considered as training or validation data.

    Returns:
        pd.DataFrame: Deduplicated DataFrame with duplicates removed where CCS difference ≥ ccs_threshold, or empty DataFrame if no data.
    """
    # Reset index at the beginning to ensure consistency
    train_val_test_df = train_val_test_df.reset_index(drop=True)

    # Print starting number of rows
    initial_rows = len(train_val_test_df)
    print(f"    ◦ Initial number of rows: {initial_rows}")

    # Identify duplicates based on id_col and adduct_col
    duplicates = train_val_test_df[train_val_test_df.duplicated(subset=[columns["id"], columns["adduct"]], keep=False)]

    # Process duplicates to remove all the compounds if, at least one, differs more than a 3% from the mean
    removed_groups = 0
    keep_indices = []
    if not duplicates.empty:
        for (_, _), group in duplicates.groupby([columns["id"], columns["adduct"]]):
            ccs_values = group[columns["target"]].values
            mean_ccs = ccs_values.mean()
            # Check if any CCS value differs by more than 3% from the mean
            if all(abs(ccs - mean_ccs) / mean_ccs <= ccs_threshold for ccs in ccs_values):
                # Keep all indices from the group
                keep_indices.extend(group.index)
            else:
                removed_groups += 1
        # Calculate total items removed (all duplicate rows minus kept rows)
        items_removed = len(duplicates) - len(keep_indices)
        print(f"    ◦ Number of items removed due to >3% CCS variation: {items_removed}")

    # Create deduplicated DataFrame using kept indices
    deduplicated_df = train_val_test_df.loc[keep_indices].copy()

    # Include non-duplicates from the original DataFrame
    non_duplicates = train_val_test_df.loc[~train_val_test_df.index.isin(duplicates.index)]
    deduplicated_df = pandas.concat([deduplicated_df, non_duplicates], ignore_index=True)

    # Reset index
    deduplicated_df = deduplicated_df.reset_index(drop=True)

    # Process remaining duplicates in deduplicated_df where dataset is in train_val_dbs
    # and keep only one from the training datasets with the average of all the training ones
    train_val_duplicates = deduplicated_df[
        (deduplicated_df['dataset'].isin(train_val_dbs)) &
        (deduplicated_df.duplicated(subset=[columns["id"], columns["adduct"]], keep=False))
        ]
    additional_removed = 0
    if not train_val_duplicates.empty:
        # Group by id_col and adduct_col, keep one row with averaged CCS
        grouped = train_val_duplicates.groupby([columns["id"], columns["adduct"]])
        keep_rows = []
        for (_, _), group in grouped:
            avg_ccs = group['ccs'].mean()
            # Keep the first row of the group, update its CCS value
            first_row = group.iloc[0].copy()
            first_row['ccs'] = avg_ccs
            keep_rows.append(first_row)
            additional_removed += len(group) - 1  # Count removed duplicates
        # Create DataFrame from kept rows
        keep_df = pandas.DataFrame(keep_rows).reset_index(drop=True)
        # Remove all train_val_duplicates from deduplicated_df
        deduplicated_df = deduplicated_df[
            ~((deduplicated_df['dataset'].isin(train_val_dbs)) &
              (deduplicated_df.duplicated(subset=[columns["id"], columns["adduct"]], keep=False)))
        ]
        # Concatenate the kept rows with averaged CCS
        deduplicated_df = pandas.concat([deduplicated_df, keep_df], ignore_index=True)
        items_removed = len(train_val_duplicates) - len(keep_df)
        print(f"    ◦ Number of items removed due to duplication in train_val: {items_removed}")
    # Reset indices one last time
    deduplicated_df = deduplicated_df.reset_index(drop=True)

    # Print summary
    final_rows = len(deduplicated_df)
    print(f"    ◦ Removed {initial_rows-final_rows} rows, {removed_groups} groups\n"
          f"    ◦ Final number of rows: {final_rows}")

    return deduplicated_df