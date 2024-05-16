import os
import importlib
from torch_geometric.datasets import MoleculeNet

class HomogeneousDataset:
    def __init__(self):
        self.dataset = {
            "QM7b": {
                "description": "The QM7b dataset from the 'MoleculeNet: A Benchmark for Molecular Machine Learning' paper, consisting of 7,211 molecules with 14 regression targets.",
                "is_molecule_data": True,
                "required_args": {"name": "QM7b"}
            },
            "QM9": {
                "description": "The QM9 dataset from the 'MoleculeNet: A Benchmark for Molecular Machine Learning' paper, consisting of about 130,000 molecules with 19 regression targets.",
                "is_molecule_data": True,
                "required_args": {"name": "None"}
            },
            "MD17": {
                "description": "A variety of ab-initio molecular dynamics trajectories from the authors of sGDML.",
                "is_molecule_data": True,
                "required_args": {"name": "aspirin"}  # Beispiel für Aspirin-Trajektorie
            },
            "ZINC": {
                "description": "The ZINC dataset from the ZINC database and the 'Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules' paper, containing about 250,000 molecular graphs with up to 38 heavy atoms.",
                "is_molecule_data": True
            },
            "AQSOL": {
                "description": "The AQSOL dataset from the Benchmarking Graph Neural Networks paper based on AqSolDB, a standardized database of 9,982 molecular graphs with their aqueous solubility values, collected from 9 different data sources.",
                "is_molecule_data": True,
                "required_args": {"name": "AQSOL"}
            },
            "MoleculeNet": {
                "description": "The MoleculeNet benchmark collection from the 'MoleculeNet: A Benchmark for Molecular Machine Learning' paper, containing datasets from physical chemistry, biophysics and physiology.",
                "is_molecule_data": True
            },
            "PCQM4Mv2": {
                "description": "The PCQM4Mv2 dataset from the 'OGB-LSC: A Large-Scale Challenge for Machine Learning on Graphs' paper.",
                "is_molecule_data": True
            },
            "HydroNet": {
                "description": "The HydroNet dataset from the 'HydroNet: Benchmark Tasks for Preserving Intermolecular Interactions and Structural Motifs in Predictive and Generative Models for Molecular Data' paper, consisting of 5 million water clusters held together by hydrogen bonding networks.",
                "is_molecule_data": True
            }
        }
