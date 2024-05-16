import os
import importlib
import torch
from tqdm import tqdm
import time
from graphviz import Digraph
from concurrent.futures import ThreadPoolExecutor, as_completed
from rdkit import Chem

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

    def _process_sample(self, index_sample):
        index, sample = index_sample
        sample_dict = {}
        for key, value in sample.items():
            if key == 'smiles':  # Assume the key for SMILES is 'smiles'
                try:
                    mol = Chem.MolFromSmiles(value)
                    if mol is not None:
                        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                        sample_dict[key] = canonical_smiles
                    else:
                        sample_dict[key] = "Invalid SMILES"
                except Exception as e:
                    sample_dict[key] = f"Error: {e}"
            else:
                sample_dict[key] = value
        return index, sample_dict

    def dataset_to_dict(self, dataset, use_multithreading=True):
        print("Funktion: TorchGeometricDatasets.dataset_to_dict")
        data_dict = {}
        num_samples = len(dataset)

        if use_multithreading:
            num_workers = min(32, num_samples)  # You can adjust the number of threads as needed
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(self._process_sample, (i, dataset[i])): i for i in range(num_samples)}
                for future in tqdm(as_completed(futures), total=num_samples, desc="Converting dataset to dict"):
                    index, sample_dict = future.result()
                    data_dict[index] = sample_dict
        else:
            for i in tqdm(range(num_samples), desc="Converting dataset to dict"):
                index, sample_dict = self._process_sample((i, dataset[i]))
                data_dict[index] = sample_dict

        return data_dict

    def download_dataset(self, nameOfDict, use_multithreading=True):
        print("Funktion: TorchGeometricDatasets.download_dataset")
        dataset_dir = self._create_dataset_dir(nameOfDict)

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        dataset_module = importlib.import_module('torch_geometric.datasets')
        dataset_class = getattr(dataset_module, nameOfDict)
        dataset = dataset_class(root=dataset_dir)
        data_dict = self.dataset_to_dict(dataset, use_multithreading)

        key = list(data_dict.keys())[0]
        dictionary = data_dict[key]
        self.DictVisulizer.dict_to_graph(dictionary, nameOfDict, path2Dataset=dataset_dir, nObjects=len(data_dict))

if __name__ == "__main__":
    start_time = time.time()

    # Beispiel für die Verwendung der Klasse
    datasets = TorchGeometricDatasets()

    nameDataset = 'QM9'
    use_multithreading = True  # Setze auf False, um Multithreading zu deaktivieren
    datasets.download_dataset(nameDataset, use_multithreading)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Benötigte Zeit: {elapsed_time:.2f} Sekunden")
