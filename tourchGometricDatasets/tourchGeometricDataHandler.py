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

# Beispiel-Daten-Dictionary
example_data = {
    'person': {
        'name': 'John',
        'age': 30,
        'city': 'New York',
        'languages': ['English', 'German'],
        'address': {
            'street': '123 Main St',
            'zip_code': '10001'
        }
    }
}
# Funktion zum Erstellen eines Graphen aus einem Dictionary
def dict_to_graph(dictionary,nameOfDict='', G=None, parent_key=None):
    if G is None:
        G = nx.DiGraph()
        parent_key = nameOfDict
    for key, value in dictionary.items():
        if key is not None and value is not None:
            current_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                G.add_node(current_key)
                G = dict_to_graph(value, G, current_key)
            elif isinstance(value, list):
                G.add_node(current_key)
                for i, v in enumerate(value):
                    G.add_node(f"{current_key}[{i}]")
                    G.add_edge(current_key, f"{current_key}[{i}]")
            else:
                G.add_node(current_key)
                G.add_edge(parent_key, current_key)
                G.nodes[current_key]['value'] = value


    graph = G
    # Drucke den Graphen
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 6))
    nx.draw(graph, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold')
    plt.title('Graph des Beispiel-Daten-Dictionary')
    plt.show()



class DictVisualization:
    def __init__(self, data):
        self.data = data
        self.graph = graphviz.Digraph()

    def _serialize_value(self, value):
        if isinstance(value, torch.Tensor):
            return value.tolist()  # Tensor in eine Liste von Zahlen umwandeln
        else:
            return value

    def _visualize_dict(self, data, parent_key=None):
        for key, value in data.items():
            key_str = str(key)  # Schlüssel in einen String umwandeln
            if parent_key is not None:
                self.graph.edge(str(parent_key), key_str)  # Elternknoten auch in einen String umwandeln
            if isinstance(value, dict):
                self._visualize_dict(value, key_str)
            else:
                serialized_value = self._serialize_value(value)
                self.graph.node(key_str, label=key_str + ': ' + json.dumps(serialized_value))  # Wert serialisieren und als JSON-String anzeigen

    def visualize(self):
        self._visualize_dict(self.data)
        self.graph.render(filename='dict_visualization', format='png', cleanup=True)
        print("Visualization saved as 'dict_visualization.png'")
class TorchGeometricDatasets:
    def __init__(self):
        self.base_dir = os.path.join(os.getcwd(), 'datasets')

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
        name = 'QM7b'
        dataset_dir = self._create_dataset_dir('QM7b')

        dataset_dir = os.path.join(os.getcwd(), 'datasets', name)

        # Erstelle das Verzeichnis, falls es nicht existiert
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        dataset_module = importlib.import_module('torch_geometric.datasets')
        dataset_class = getattr(dataset_module, name)
        dataset = dataset_class(root=dataset_dir)
        data_dict = self.dataset_to_dict(dataset)
        print(f"{name} data_dict: {data_dict.keys()}")

        # Baumdarstellung ausgeben
        # Visualisierung erstellen
        key = list(data_dict.keys())[0]
        dict_to_graph(data_dict[key])
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
