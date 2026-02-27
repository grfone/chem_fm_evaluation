import os
os.environ["KERAS_BACKEND"] = "torch"

from src.FileManager import ExtractFiles, UnifyFormats, PrepareDataset, UnifyEmbeddings
from src.Encoders import GenerateEmbeddings
from src.CCSTrainer import CCSTrainer
from src.Statistics import GlobalStats
from src.Visualization import Umaps


if __name__ == "__main__":
    print(f"Environment variable KERAS_BACKEND = {os.environ.get('KERAS_BACKEND')}")

    ExtractFiles()
    UnifyFormats()
    PrepareDataset()
    GenerateEmbeddings()
    UnifyEmbeddings()
    CCSTrainer()
    GlobalStats()
    Umaps()

