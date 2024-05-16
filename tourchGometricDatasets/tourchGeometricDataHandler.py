import os
import importlib
import torch
from tqdm import tqdm
import time
from graphviz import Digraph
from rdkit import Chem
from rdkit import RDLogger
import sqlite3
from molvs import standardize_smiles
# Deactivate RDKit warnings
#RDLogger.DisableLog('rdApp.*')

class TreeNode:
    def __init__(self, name, value=None, shape=None, dtype=None):
        self.name = name
        self.value = value if not isinstance(value, str) else value[:50]  # Show first 50 characters for strings
        self.shape = shape
        self.dtype = dtype
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self, level=0):
        ret = "\t" * level + f"{self.name} (shape={self.shape}, dtype={self.dtype}): {self.value}\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

class ObjectTree:
    def __init__(self, obj, name="root", nObjects=None):
        print("Funktion: ObjectTree.__init__")
        self.root = TreeNode(name, shape=nObjects)
        self._build_tree(obj, self.root)

    def _build_tree(self, obj, tree_node):
        if isinstance(obj, dict):
            for key, value in obj.items():
                shape, dtype = self._get_shape_and_dtype(value)
                child_node = TreeNode(key, shape=shape, dtype=dtype, value='')
                tree_node.add_child(child_node)
                self._build_tree(value, child_node)
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                shape, dtype = self._get_shape_and_dtype(item)
                child_node = TreeNode(f"index_{index}", shape=shape, dtype=dtype)
                tree_node.add_child(child_node)
                self._build_tree(item, child_node)
        elif isinstance(obj, str):
            shape, dtype = self._get_shape_and_dtype(obj)
            tree_node.shape = shape
            tree_node.dtype = dtype
            tree_node.value = obj
        else:
            shape, dtype = self._get_shape_and_dtype(obj)
            tree_node.shape = shape
            tree_node.dtype = dtype
            tree_node.value = obj

    def _get_shape_and_dtype(self, value):
        if isinstance(value, torch.Tensor):
            return tuple(value.shape), value.dtype
        elif isinstance(value, (list, dict)):
            return len(value), type(value)
        else:
            return None, type(value)

    def __repr__(self):
        return repr(self.root)

    def plot(self, name, dataset_dir=None):
        filename = f"{name}_object_tree"
        filePath = os.path.join(dataset_dir, filename) if dataset_dir else filename

        dot = Digraph()
        dot.attr(label=f"Root: {self.root.shape} objects")
        self._add_nodes(dot, self.root)
        dot.render(filePath, format='svg', view=False)
        self._save_txt(filePath)
        print(f"Saved SVG to {filePath}.svg")
        print(f"Saved TXT to {filePath}.txt")

    def _add_nodes(self, dot, node, parent_id=None):
        node_id = str(id(node))
        label = f"{node.name}\n(shape={node.shape}, dtype={node.dtype})\n{node.value}"
        dot.node(node_id, label)
        if parent_id:
            dot.edge(parent_id, node_id)
        for child in node.children:
            self._add_nodes(dot, child, node_id)

    def _save_txt(self, filename):
        with open(f"{filename}.txt", "w") as file:
            file.write(self.__repr__())




class DictVisualization:
    def dict_to_graph(self, dictionary, name, path2Dataset=None, nObjects=None):
        print("Funktion: DictVisualization.dict_to_graph")
        tree = ObjectTree(dictionary, name, nObjects=nObjects)
        nameTreeDir = os.path.join(path2Dataset, 'tree')
        os.makedirs(nameTreeDir, exist_ok=True)
        tree.plot(name, nameTreeDir)


class TorchGeometricDatasets:
    def __init__(self):
        print("Funktion: TorchGeometricDatasets.__init__")
        self.base_dir = os.path.join(os.getcwd(), 'datasets')
        self.DictVisulizer = DictVisualization()

    def _create_dataset_dir(self, name):
        print("Funktion: TorchGeometricDatasets._create_dataset_dir")
        dataset_dir = os.path.join(self.base_dir, name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        return dataset_dir

    def _process_sample(self, index_sample, doPrint=False, showWarnings=False):
        index, sample = index_sample
        sample_dict = {}
        failed = False
        oldSmile = None

        if not showWarnings:
            RDLogger.DisableLog('rdApp.*')

        for key, value in sample.items():
            if key == 'smiles':  # Assume the key for SMILES is 'smiles'
                try:
                    oldSmile = value

                    if doPrint:
                        print(f"oldSmile: {oldSmile}")
                    value = standardize_smiles(value)
                    mol = Chem.MolFromSmiles(value)

                    if mol is not None:
                        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                        sample_dict[key] = value
                        sample_dict['canonical_smiles'] = canonical_smiles

                    else:
                        sample_dict[key] = "Invalid SMILES"

                except Exception as e:
                    failed = True
                    if doPrint:
                        print(f"Error processing SMILES: {e}")
                        print(f"Sample index: {index}")
                        print(f"Sample SMILES: {value}")
                        print(f"canonical_smiles: {canonical_smiles}")

                    if 'Explicit valence' in str(e):
                        sample_dict[key] = "Invalid SMILES"
                    else:
                        sample_dict[key] = f"Error: {e}"
            else:
                sample_dict[key] = value

        sample_dict['has_failed'] = failed
        return index, sample_dict, failed, oldSmile

    def dataset_to_dict(self, dataset):
        print("Funktion: TorchGeometricDatasets.dataset_to_dict")
        data_dict = {}
        num_samples = len(dataset)

        nSuccess = 0
        nFail = 0
        smilesSuccess = []
        smilesFail = []
        doPrint = False
        showWarnings = False

        if not showWarnings:
            print("Warnings are disabled")
        for i in tqdm(range(num_samples), desc="Converting dataset to dict"):
            index, sample_dict, failed, oldSmile= self._process_sample((i, dataset[i]), doPrint, showWarnings)
            data_dict[index] = sample_dict

            if failed:
                nFail += 1
                smilesFail.append(sample_dict['smiles'])
            else:
                nSuccess += 1
                smilesSuccess.append(sample_dict['smiles'])

        print(f"failed SMILES: {smilesFail}")
        print(f"nSuccess {nSuccess} SMILES processed successfully, failed {nFail} SMILES processing")

        return data_dict

    def download_dataset(self, nameOfDict):
        print("Funktion: TorchGeometricDatasets.download_dataset")
        dataset_dir = self._create_dataset_dir(nameOfDict)

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        dataset_module = importlib.import_module('torch_geometric.datasets')
        dataset_class = getattr(dataset_module, nameOfDict)
        dataset = dataset_class(root=dataset_dir)
        data_dict = self.dataset_to_dict(dataset)

        key = list(data_dict.keys())[0]
        dictionary = data_dict[key]
        self.DictVisulizer.dict_to_graph(dictionary, nameOfDict, path2Dataset=dataset_dir, nObjects=len(data_dict))
        return data_dict

if __name__ == "__main__":
    start_time = time.time()

    # Beispiel für die Verwendung der Klasse
    datasets = TorchGeometricDatasets()

    for nameDataset in tqdm(['QM9', 'ZINC', 'AQSOL', 'MoleculeNet', 'PCQM4Mv2', 'HydroNet']):
        print('nameDataset', nameDataset)
        data_dict = datasets.download_dataset(nameDataset)

        data_dict_qm9 = datasets.download_dataset(nameDataset)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Benötigte Zeit: {elapsed_time:.2f} Sekunden")
