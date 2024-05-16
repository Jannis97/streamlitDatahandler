import os
import importlib
import torch
from tqdm import tqdm
import time
from graphviz import Digraph

class TreeNode:
    def __init__(self, name, value=None, shape=None, dtype=None):
        self.name = name
        self.value = value if not isinstance(value, str) else value[:30]  # Show first 30 characters for strings
        self.shape = shape
        self.dtype = dtype
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self, level=0):
        ret = "\t" * level + f"{repr(self.name)} (shape={self.shape}, dtype={self.dtype}): {repr(self.value)}\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret
class ObjectTree:
    def __init__(self, obj, name="root"):
        nObjects = len(obj) if isinstance(obj, (list, dict)) else 1
        self.root = TreeNode(name, shape=nObjects)
        self._build_tree(obj, self.root)

    def _build_tree(self, obj, tree_node):
        if isinstance(obj, dict):
            for key, value in obj.items():
                shape, dtype = self._get_shape_and_dtype(value)
                child_node = TreeNode(key, shape=shape, dtype=dtype)
                tree_node.add_child(child_node)
                self._build_tree(value, child_node)
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                shape, dtype = self._get_shape_and_dtype(item)
                child_node = TreeNode(f"index_{index}", shape=shape, dtype=dtype)
                tree_node.add_child(child_node)
                self._build_tree(item, child_node)
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
        self._add_nodes(dot, self.root)
        dot.render(filePath, format='svg', view=False)
        self._save_txt(filename)

    def _add_nodes(self, dot, node, parent_id=None):
        node_id = str(id(node))
        label = f"{node.name}\n(shape={node.shape}, dtype={node.dtype})"
        dot.node(node_id, label)
        if parent_id:
            dot.edge(parent_id, node_id)
        for child in node.children:
            self._add_nodes(dot, child, node_id)

    def _save_txt(self, filename):
        with open(f"{filename}.txt", "w") as file:
            file.write(self.__repr__())

class DictVisualization:
    def dict_to_graph(self, dictionary, name, path2Dataset=None):
        tree = ObjectTree(dictionary, name)
        nameTreeDir = os.path.join(path2Dataset, 'tree')
        tree.plot(name, nameTreeDir)

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
        for i in tqdm(range(len(dataset)), desc="Converting dataset to dict"):
            sample = dataset[i]
            sample_dict = {}
            for key, value in sample.items():
                sample_dict[key] = value
            data_dict[i] = sample_dict
        return data_dict

    def download_QM7b(self, nameOfDict):

        dataset_dir = self._create_dataset_dir(nameOfDict)

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        dataset_module = importlib.import_module('torch_geometric.datasets')
        dataset_class = getattr(dataset_module, nameOfDict)
        dataset = dataset_class(root=dataset_dir)
        data_dict = self.dataset_to_dict(dataset)
        print(f"{nameOfDict} data_dict: {data_dict.keys()}")

        key = list(data_dict.keys())[0]
        dictionary = data_dict[key]
        self.DictVisulizer.dict_to_graph(dictionary, nameOfDict, path2Dataset=dataset_dir)


if __name__ == "__main__":
    start_time = time.time()

    # Beispiel für die Verwendung der Klasse
    datasets = TorchGeometricDatasets()

    nameDataset = 'QM9'
    datasets.download_QM7b(nameDataset)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Benötigte Zeit: {elapsed_time:.2f} Sekunden")