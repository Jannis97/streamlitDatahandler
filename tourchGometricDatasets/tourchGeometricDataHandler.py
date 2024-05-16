# dataset_printer.py
import os
from tourtchGeometricDatasetDictHomogenous import homogeneous_dataset
import importlib
import torch_geometric
class DatasetSelectior:
    def __init__(self, datasetAll):
        self.datasetAll = datasetAll

        molDatasets = []
        notMolDatasets = []
        molecule_datasets = datasetAll.keys()
        for name in molecule_datasets:
            is_molecule_data = datasetAll[name]['is_molecule_data']

            if is_molecule_data:
                molDatasets.append(name)
            else:
                notMolDatasets.append(name)

        print(f"molDatasets: {molDatasets}")
        print(f"notMolDatasets: {notMolDatasets}")

        self.molDatasets = molDatasets
        self.notMolDatasets = notMolDatasets

    def select_dataset(self, name):
        if name in self.datasetAll:
            dataset_info = self.datasetAll[name]
            dataset_dir = os.path.join(os.getcwd(), 'datasets', name)

            # Erstelle das Verzeichnis, falls es nicht existiert
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            dataset_module = importlib.import_module('torch_geometric.datasets')

            dataset_class = getattr(dataset_module, name)
            dataset = dataset_class(root=dataset_dir)

            num_elements = len(dataset)
            print(f"Der Datensatz '{name}' wurde erfolgreich geladen und enthält {num_elements} Elemente.")
            return dataset

# Beispielverwendung
if __name__ == "__main__":
    datasetAll = homogeneous_dataset().dataset
    dataset_selector = DatasetSelectior(datasetAll)
    datasetLoaded = dataset_selector.select_dataset(name ='QM9')
