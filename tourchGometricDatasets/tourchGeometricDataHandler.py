import os
import sqlite3
import importlib
import subprocess
import sys
import graphviz
import json
import torch
import networkx as nx
import matplotlib.pyplot as plt
class DictVisualization:
    def __init__(self):
        pass

    def dict_to_graph(self, dictionary, nameOfDict='', G=None, parent_key=None):
        if G is None:
            G = nx.DiGraph()
            parent_key = nameOfDict
            G.add_node(parent_key)  # Füge die Wurzel als ersten Knoten hinzu
        for key, value in dictionary.items():
            if key is not None and value is not None:
                current_key = f"{parent_key}.{key}" if parent_key else key
                if isinstance(value, dict):
                    G.add_node(current_key)
                    G.add_edge(parent_key, current_key)  # Kante von der Wurzel zum aktuellen Knoten hinzufügen
                    G = self.dict_to_graph(value, nameOfDict, G, current_key)
                elif isinstance(value, list):
                    G.add_node(current_key)
                    for i, v in enumerate(value):
                        G.add_node(f"{current_key}[{i}]")
                        G.add_edge(current_key, f"{current_key}[{i}]")
                else:
                    G.add_node(current_key)
                    G.add_edge(parent_key, current_key)
                    valueType = type(value).__name__
                    valueDtype = value.dtype if isinstance(value, torch.Tensor) else None

                    G.nodes[current_key]['value'] = f"valueType:({valueType}), valueDtype:({valueDtype})"

        # Layout für den Graphen festlegen
        pos = nx.spring_layout(G, seed=42)
        # Vertikale Ausrichtung der Knoten anpassen
        pos_labels = {node: (pos[node][0], pos[node][1] * -1) for node in G.nodes()}

        # Graph anzeigen
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos_labels, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold', verticalalignment='bottom')
        plt.title(f'Graph des {nameOfDict} Beispiel-Daten-Dictionary (Baumdarstellung)')
        plt.show()



class TorchGeometricDatasets:
    def __init__(self):
        self.base_dir = os.path.join(os.getcwd(), 'datasets')

        self.DictVisulizer = DictVisualization()

    def _create_dataset_dir(self, name):
        dataset_dir = os.path.join(self.base_dir, name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        return dataset_dir

    def dataset_to_dict(self, dataset):
        data_dict = {}
        for i in range(len(dataset)):
            sample = dataset[i]
            sample_dict = {}
            # Annahme: Jedes Sample im Dataset ist ein Dictionary
            for key, value in sample.items():
                # Hier kannst du die Werte deines Samples anpassen, falls erforderlich
                sample_dict[key] = value
            data_dict[i] = sample_dict
        return data_dict

    def download_QM7b(self):
        nameOfDict = 'QM7b'
        dataset_dir = self._create_dataset_dir(nameOfDict)

        dataset_dir = os.path.join(os.getcwd(), 'datasets', nameOfDict)

        # Erstelle das Verzeichnis, falls es nicht existiert
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        dataset_module = importlib.import_module('torch_geometric.datasets')
        dataset_class = getattr(dataset_module, nameOfDict)
        dataset = dataset_class(root=dataset_dir)
        data_dict = self.dataset_to_dict(dataset)
        print(f"{nameOfDict} data_dict: {data_dict.keys()}")

        # Baumdarstellung ausgeben
        # Visualisierung erstellen

        key = list(data_dict.keys())[0]
        dictionary = data_dict[key]
        self.DictVisulizer.dict_to_graph(dictionary,nameOfDict, G=None, parent_key=None)
        a = 1
    def download_QM9(self):
        dataset_dir = self._create_dataset_dir('QM9')
        # Logik zum Herunterladen und Installieren von QM9 hier
        print(f"QM9 Datensatz installiert in {dataset_dir}")

    def download_MD17(self):
        dataset_dir = self._create_dataset_dir('MD17')
        # Logik zum Herunterladen und Installieren von MD17 hier
        print(f"MD17 Datensatz installiert in {dataset_dir}")

    def download_ZINC(self):
        dataset_dir = self._create_dataset_dir('ZINC')
        # Logik zum Herunterladen und Installieren von ZINC hier
        print(f"ZINC Datensatz installiert in {dataset_dir}")

    def download_AQSOL(self):
        dataset_dir = self._create_dataset_dir('AQSOL')
        # Logik zum Herunterladen und Installieren von AQSOL hier
        print(f"AQSOL Datensatz installiert in {dataset_dir}")

    def download_MoleculeNet(self):
        dataset_dir = self._create_dataset_dir('MoleculeNet')
        # Logik zum Herunterladen und Installieren von MoleculeNet hier
        print(f"MoleculeNet Datensatz installiert in {dataset_dir}")

    def download_PCQM4Mv2(self):
        dataset_dir = self._create_dataset_dir('PCQM4Mv2')
        # Logik zum Herunterladen und Installieren von PCQM4Mv2 hier
        print(f"PCQM4Mv2 Datensatz installiert in {dataset_dir}")

    def download_HydroNet(self):
        dataset_dir = self._create_dataset_dir('HydroNet')
        # Logik zum Herunterladen und Installieren von HydroNet hier
        print(f"HydroNet Datensatz installiert in {dataset_dir}")




# Beispiel für die Verwendung der Klasse
datasets = TorchGeometricDatasets()
datasets.download_QM7b()
datasets.download_QM9()
datasets.download_MD17()
datasets.download_ZINC()
datasets.download_AQSOL()
datasets.download_MoleculeNet()
datasets.download_PCQM4Mv2()
datasets.download_HydroNet()
