import numpy as np
import pandas as pd
import os
import umap
import matplotlib.pyplot as plt


class Umaps:
    """
    A class for generating and visualizing UMAP projections for different molecular embeddings.

    This class handles multiple types of molecular embeddings (ChemBERTa, fingerprints, MolFormer,
    MolMIM, UniMol) and creates UMAP visualizations with consistent class labeling and color schemes.
    It processes embeddings from CSV files, merges them with class information, computes UMAP
    projections with different metrics and parameters, and generates publication-quality plots.

    Attributes:
        file_embeddings_list (list): List of tuples containing embedding file information
        classes_priority (list): Ordered list of chemical classes for consistent labeling
        df_cls (pandas.DataFrame): DataFrame containing InChI keys and their representative class labels
    """

    def __init__(self):
        """
        Initialize the Umaps class and process all embedding files.

        Sets up the embedding file configurations, creates output directories, defines class
        priority ordering, loads and preprocesses class labels, identifies common InChI keys
        across all datasets, and processes each embedding file to generate UMAP visualizations.
        """

        self.results_subfolder = "results/umaps"

        # Return if done:
        if os.path.exists(self.results_subfolder):
            return
        else:
            os.makedirs(self.results_subfolder, exist_ok=True)

        self.file_embeddings_list = [
            ("all_datasets_chemberta_unified.csv", "chemberta_", "chemberta"),
            ("all_datasets_fingerprints_unified.csv", "V", "fingerprints"),
            ("all_datasets_molformer_unified.csv", "molformer_", "molformer"),
            ("all_datasets_molmim_unified.csv", "molmim_", "molmim"),
            ("all_datasets_unimol_unified.csv", "unimol_", "unimol"),
        ]

        self.classes_priority = [
            "Amino acids, peptides, and analogues",
            "Peptides",
            "Fatty Acyls",
            "Glycerolipids",
            "Glycerophospholipids",
            "Cholines",
            "Amines",
            "Trialkylamines",
            "Sphingolipids",
            "Organoheterocyclic compounds",
            "Benzenoids",
            "Diazines",
            "Azoles",
            "Steroids and steroid derivatives",
            "Prenol lipids",
            "Triterpenoids",
            "Bile acids, alcohols and derivatives",
            "Phenylpropanoids and polyketides",
            "Carbohydrates and carbohydrate conjugates",
            "Organosulfur compounds",
            "Other"
        ]
        df_cls_raw = pd.read_csv("resources/all_datasets_classification.csv")

        # Get all columns that start with 'class_'
        class_columns = [col for col in df_cls_raw.columns if col.startswith('class_')]

        # Melt to combine all class columns into one, then group by inchi
        melted = df_cls_raw.melt(id_vars=['inchi'], value_vars=class_columns, value_name='class_value')

        # Get unique class values for each inchi
        unique_classes = melted.groupby('inchi')['class_value'].unique()

        # Apply the selection function
        representative = unique_classes.apply(self.select_representative)

        self.df_cls = pd.DataFrame({'inchi': representative.index, 'representative_class': representative.values})

        common_inchi = set(self.df_cls['inchi'])
        for file, _, _ in self.file_embeddings_list:
            df_emb = pd.read_csv(os.path.join("resources", file))
            df_emb = df_emb.drop_duplicates(subset=['inchi'])
            common_inchi = common_inchi.intersection(set(df_emb['inchi']))

        self.df_cls = self.df_cls[self.df_cls['inchi'].isin(common_inchi)]

        for file, prefix, name in self.file_embeddings_list:
            self.process_file(file, prefix, name)
        print("All plots saved.")

    def select_representative(self, classes_list):
        """
        Select a representative class label from a list of potential class assignments.

        For molecules with multiple possible class assignments, this method selects the most
        appropriate single class based on the predefined priority order. Classes not in the
        priority list are assigned as "Other".

        Args:
            classes_list (list): List of class labels assigned to a molecule

        Returns:
            str: A single representative class label for the molecule
        """

        # Convert to list and remove NaN values
        classes_list = [c for c in classes_list if pd.notna(c)]

        if not classes_list:
            return "Other"

        # Filter to classes in priority list
        valid_classes = [c for c in classes_list if c in self.classes_priority]

        if not valid_classes:
            return "Other"

        # Return highest priority class
        return min(valid_classes, key=lambda c: self.classes_priority.index(c))

    def process_file(self, file, prefix, name):
        """
        Process a single embedding file to generate UMAP visualizations.

        Loads the embedding file, merges it with class labels, extracts embedding vectors,
        and computes UMAP projections with appropriate metrics for the embedding type.
        Saves class distribution statistics and generates UMAP plots.

        Args:
            file (str): Name of the embedding CSV file
            prefix (str): Column prefix for embedding vectors in the CSV file
            name (str): Short name identifier for the embedding type (e.g., "chemberta")

        Note:
            For fingerprint embeddings, uses Jaccard and Rogerstanimoto metrics.
            For other embeddings, uses cosine and Euclidean metrics.
        """

        df_emb = pd.read_csv(os.path.join("resources", file))
        df_emb = df_emb.drop_duplicates(subset=["inchi"])
        df_emb = df_emb[df_emb["inchi"].isin(self.df_cls["inchi"])]

        df = pd.merge(self.df_cls, df_emb, on="inchi", how="inner")

        X = df.filter(like=prefix).values
        y = df["representative_class"].values

        # classes ordered by global priority, only those present
        classes = [c for c in self.classes_priority if c in set(y)]

        cmap = plt.get_cmap("tab20c")
        configs = ([("jaccard", 50, 0.01), ("rogerstanimoto", 50, 0.01)] if "fingerprints" in file else
                   [("cosine", 50, 0.01), ("euclidean", 50, 0.01)])

        for metric, n, d in configs:
            self.compute_umap_and_plot(X, y, classes, cmap, name, metric, n, d)

            # save class distribution
            unique, counts = np.unique(y, return_counts=True)
            dist = dict(zip(unique, counts))
            class_distribution_filename = f"umap_class_distribution_{name}_{metric}_n{n}_d{d}.txt"
            class_distribution_path = os.path.join(self.results_subfolder, class_distribution_filename)
            # Save as CSV text file
            df = pd.DataFrame(list(dist.items()), columns=['class', 'count'])
            df.to_csv(class_distribution_path, index=False)

    def compute_umap_and_plot(self, X, y, classes, cmap, name, metric, n, d):
        emb2d = umap.UMAP(n_neighbors=n, min_dist=d, metric=metric, random_state=42).fit_transform(X)
        embeddings_filename = f"umap_embeddings_{name}_{metric}_n{n}_d{d}.npy"
        embeddings_path = os.path.join(self.results_subfolder, embeddings_filename)
        np.save(embeddings_path, emb2d)
        for s in [0.01, 0.001]:
            plt.figure(figsize=(10, 8))
            for i, cls in enumerate(classes):
                plt.scatter(emb2d[y == cls, 0], emb2d[y == cls, 1], s=s, color=cmap(i), label=cls)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, markerscale=(100 / s) ** 0.5)
            plt.title(f"{name} | {metric} | n_neighbors={n} | min_dist={d}")
            plt.xlabel("UMAP-1");
            plt.ylabel("UMAP-2")
            plt.tight_layout()
            plot_filename = f"umap_{name}_{metric}_n{n}_d{d}_s{s}.png"
            plot_path = os.path.join(self.results_subfolder, plot_filename)
            plt.savefig(plot_path, dpi=400, bbox_inches='tight')
            plt.close()
