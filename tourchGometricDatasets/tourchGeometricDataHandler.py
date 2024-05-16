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
import os
import importlib
from graphviz import Digraph

class TreeNode:
    def __init__(self, name, value=None):
        self.name = name
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.name) + ": " + repr(self.value) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

class ObjectTree:
    def __init__(self, obj, name="root"):
        self.root = TreeNode(name)
        self._build_tree(obj, self.root)

    def _build_tree(self, obj, tree_node):
        if isinstance(obj, dict):
            for key, value in obj.items():
                child_node = TreeNode(key)
                tree_node.add_child(child_node)
                self._build_tree(value, child_node)
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                child_node = TreeNode(f"index_{index}")
                tree_node.add_child(child_node)
                self._build_tree(item, child_node)
        else:
            tree_node.value = obj

    def __repr__(self):
        return repr(self.root)

    def plot(self, filename="object_tree"):
        dot = Digraph()
        self._add_nodes(dot, self.root)
        dot.render(filename, format='png', view=True)

    def _add_nodes(self, dot, node, parent_id=None):
        node_id = str(id(node))
        label = f"{node.name}: {node.value}" if node.value is not None else node.name
        dot.node(node_id, label)
        if parent_id:
            dot.edge(parent_id, node_id)
        for child in node.children:
            self._add_nodes(dot, child, node_id)
class DictVisualization:
    def dict_to_graph(self, dictionary, name, G=None, parent_key=None):
        tree = ObjectTree(dictionary, name)
        tree.plot(name)


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
        nameOfDict = 'QM9'
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
        self.DictVisulizer.dict_to_graph(dictionary, nameOfDict)
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
