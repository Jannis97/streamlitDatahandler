class homogeneous_dataset:
    def __init__(self):
        self.dataset = {
        "KarateClub": {
            "description": "Zachary's karate club network from the 'An Information Flow Model for Conflict and Fission in Small Groups' paper, containing 34 nodes, connected by 156 (undirected and unweighted) edges.",
            "is_molecule_data": False
        },
        "TUDataset": {
            "description": "A variety of graph kernel benchmark datasets, e.g., 'IMDB-BINARY', 'REDDIT-BINARY' or 'PROTEINS', collected from the TU Dortmund University.",
            "is_molecule_data": False
        },
        "GNNBenchmarkDataset": {
            "description": "A variety of artificially and semi-artificially generated graph datasets from the 'Benchmarking Graph Neural Networks' paper.",
            "is_molecule_data": False
        },
        "Planetoid": {
            "description": "The citation network datasets 'Cora', 'CiteSeer' and 'PubMed' from the 'Revisiting Semi-Supervised Learning with Graph Embeddings' paper.",
            "is_molecule_data": False
        },
        "NELL": {
            "description": "The NELL dataset, a knowledge graph from the 'Toward an Architecture for Never-Ending Language Learning' paper.",
            "is_molecule_data": False
        },
        "CitationFull": {
            "description": "The full citation network datasets from the 'Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking' paper.",
            "is_molecule_data": False
        },
        "CoraFull": {
            "description": "Alias for CitationFull with name='Cora'.",
            "is_molecule_data": False
        },
        "Coauthor": {
            "description": "The Coauthor CS and Coauthor Physics networks from the 'Pitfalls of Graph Neural Network Evaluation' paper.",
            "is_molecule_data": False
        },
        "Amazon": {
            "description": "The Amazon Computers and Amazon Photo networks from the 'Pitfalls of Graph Neural Network Evaluation' paper.",
            "is_molecule_data": False
        },
        "PPI": {
            "description": "The protein-protein interaction networks from the 'Predicting Multicellular Function through Multi-layer Tissue Networks' paper, containing positional gene sets, motif gene sets and immunological signatures as features (50 in total) and gene ontology sets as labels (121 in total).",
            "is_molecule_data": False
        },
        "Reddit": {
            "description": "The Reddit dataset from the 'Inductive Representation Learning on Large Graphs' paper, containing Reddit posts belonging to different communities.",
            "is_molecule_data": False
        },
        "Reddit2": {
            "description": "The Reddit dataset from the 'GraphSAINT: Graph Sampling Based Inductive Learning Method' paper, containing Reddit posts belonging to different communities.",
            "is_molecule_data": False
        },
        "Flickr": {
            "description": "The Flickr dataset from the 'GraphSAINT: Graph Sampling Based Inductive Learning Method' paper, containing descriptions and common properties of images.",
            "is_molecule_data": False
        },
        "Yelp": {
            "description": "The Yelp dataset from the 'GraphSAINT: Graph Sampling Based Inductive Learning Method' paper, containing customer reviewers and their friendship.",
            "is_molecule_data": False
        },
        "AmazonProducts": {
            "description": "The Amazon dataset from the 'GraphSAINT: Graph Sampling Based Inductive Learning Method' paper, containing products and its categories.",
            "is_molecule_data": False
        },
        "QM7b": {
            "description": "The QM7b dataset from the 'MoleculeNet: A Benchmark for Molecular Machine Learning' paper, consisting of 7,211 molecules with 14 regression targets.",
            "is_molecule_data": True
        },
        "QM9": {
            "description": "The QM9 dataset from the 'MoleculeNet: A Benchmark for Molecular Machine Learning' paper, consisting of about 130,000 molecules with 19 regression targets.",
            "is_molecule_data": True
        },
        "MD17": {
            "description": "A variety of ab-initio molecular dynamics trajectories from the authors of sGDML.",
            "is_molecule_data": True
        },
        "ZINC": {
            "description": "The ZINC dataset from the ZINC database and the 'Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules' paper, containing about 250,000 molecular graphs with up to 38 heavy atoms.",
            "is_molecule_data": True
        },
        "AQSOL": {
            "description": "The AQSOL dataset from the Benchmarking Graph Neural Networks paper based on AqSolDB, a standardized database of 9,982 molecular graphs with their aqueous solubility values, collected from 9 different data sources.",
            "is_molecule_data": True
        },
        "MoleculeNet": {
            "description": "The MoleculeNet benchmark collection from the 'MoleculeNet: A Benchmark for Molecular Machine Learning' paper, containing datasets from physical chemistry, biophysics and physiology.",
            "is_molecule_data": True
        },
        "PCQM4Mv2": {
            "description": "The PCQM4Mv2 dataset from the 'OGB-LSC: A Large-Scale Challenge for Machine Learning on Graphs' paper.",
            "is_molecule_data": True
        },
        "Entities": {
            "description": "The relational entities networks 'AIFB', 'MUTAG', 'BGS' and 'AM' from the 'Modeling Relational Data with Graph Convolutional Networks' paper.",
            "is_molecule_data": False
        },
        "RelLinkPredDataset": {
            "description": "The relational link prediction datasets from the 'Modeling Relational Data with Graph Convolutional Networks' paper.",
            "is_molecule_data": False
        },
        "GEDDataset": {
            "description": "The GED datasets from the 'Graph Edit Distance Computation via Graph Neural Networks' paper.",
            "is_molecule_data": False
        },
        "AttributedGraphDataset": {
            "description": "A variety of attributed graph datasets from the 'Scaling Attributed Network Embedding to Massive Graphs' paper.",
            "is_molecule_data": False
        },
        "MNISTSuperpixels": {
            "description": "MNIST superpixels dataset from the 'Geometric Deep Learning on Graphs and Manifolds Using Mixture Model CNNs' paper, containing 70,000 graphs with 75 nodes each.",
            "is_molecule_data": False
        },
        "FAUST": {
            "description": "The FAUST humans dataset from the 'FAUST: Dataset and Evaluation for 3D Mesh Registration' paper, containing 100 watertight meshes representing 10 different poses for 10 different subjects.",
            "is_molecule_data": False
        },
        "DynamicFAUST": {
            "description": "The dynamic FAUST humans dataset from the 'Dynamic FAUST: Registering Human Bodies in Motion' paper.",
            "is_molecule_data": False
        },
        "ShapeNet": {
            "description": "The ShapeNet part level segmentation dataset from the 'A Scalable Active Framework for Region Annotation in 3D Shape Collections' paper, containing about 17,000 3D shape point clouds from 16 shape categories.",
            "is_molecule_data": False
        },
        "ModelNet": {
            "description": "The ModelNet10/40 datasets from the '3D ShapeNets: A Deep Representation for Volumetric Shapes' paper, containing CAD models of 10 and 40 categories, respectively.",
            "is_molecule_data": False
        },
        "CoMA": {
            "description": "The CoMA 3D faces dataset from the 'Generating 3D faces using Convolutional Mesh Autoencoders' paper, containing 20,466 meshes of extreme expressions captured over 12 different subjects.",
            "is_molecule_data": False
        },
        "SHREC2016": {
            "description": "The SHREC 2016 partial matching dataset from the 'SHREC'16: Partial Matching of Deformable Shapes' paper.",
            "is_molecule_data": False
        },
        "TOSCA": {
            "description": "The TOSCA dataset from the 'Numerical Geometry of Non-Rigid Shapes' book, containing 80 meshes.",
            "is_molecule_data": False
        },
        "PCPNetDataset": {
            "description": "The PCPNet dataset from the 'PCPNet: Learning Local Shape Properties from Raw Point Clouds' paper, consisting of 30 shapes, each given as a point cloud, densely sampled with 100k points.",
            "is_molecule_data": False
        },
        "S3DIS": {
            "description": "The (pre-processed) Stanford Large-Scale 3D Indoor Spaces dataset from the '3D Semantic Parsing of Large-Scale Indoor Spaces' paper, containing point clouds of six large-scale indoor parts in three buildings with 12 semantic elements (and one clutter class).",
            "is_molecule_data": False
        },
        "GeometricShapes": {
            "description": "Synthetic dataset of various geometric shapes like cubes, spheres or pyramids.",
            "is_molecule_data": False
        },
        "BitcoinOTC": {
            "description": "The Bitcoin-OTC dataset from the 'EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs' paper, consisting of 138 who-trusts-whom networks of sequential time steps.",
            "is_molecule_data": False
        },
        "GDELTLite": {
            "description": "The (reduced) version of the Global Database of Events, Language, and Tone (GDELT) dataset used in the 'Do We Really Need Complicated Model Architectures for Temporal Networks?' paper, consisting of events collected from 2016 to 2020.",
            "is_molecule_data": False
        },
        "ICEWS18": {
            "description": "The Integrated Crisis Early Warning System (ICEWS) dataset used in the, e.g., 'Recurrent Event Network for Reasoning over Temporal Knowledge Graphs' paper, consisting of events collected from 1/1/2018 to 10/31/2018 (24 hours time granularity).",
            "is_molecule_data": False
        },
        "GDELT": {
            "description": "The Global Database of Events, Language, and Tone (GDELT) dataset used in the, e.g., 'Recurrent Event Network for Reasoning over Temporal Knowledge Graphs' paper, consisting of events collected from 1/1/2018 to 1/31/2018 (15 minutes time granularity).",
            "is_molecule_data": False
        },
        "WILLOWObjectClass": {
            "description": "The WILLOW-ObjectClass dataset from the 'Learning Graphs to Match' paper, containing 10 equal keypoints of at least 40 images in each category.",
            "is_molecule_data": False
        },
        "PascalVOCKeypoints": {
            "description": "The Pascal VOC 2011 dataset with Berkeley annotations of keypoints from the 'Poselets: Body Part Detectors Trained Using 3D Human Pose Annotations' paper, containing 0 to 23 keypoints per example over 20 categories.",
            "is_molecule_data": False
        },
        "PascalPF": {
            "description": "The Pascal-PF dataset from the 'Proposal Flow' paper, containing 4 to 16 keypoints per example over 20 categories.",
            "is_molecule_data": False
        },
        "SNAPDataset": {
            "description": "A variety of graph datasets collected from SNAP at Stanford University.",
            "is_molecule_data": False
        },
        "SuiteSparseMatrixCollection": {
            "description": "A suite of sparse matrix benchmarks known as the Suite Sparse Matrix Collection collected from a wide range of applications.",
            "is_molecule_data": False
        },
        "WordNet18": {
            "description": "The WordNet18 dataset from the 'Translating Embeddings for Modeling Multi-Relational Data' paper, containing 40,943 entities, 18 relations and 151,442 fact triplets, e.g., furniture includes bed.",
            "is_molecule_data": False
        },
        "WordNet18RR": {
            "description": "The WordNet18RR dataset from the 'Convolutional 2D Knowledge Graph Embeddings' paper, containing 40,943 entities, 11 relations and 93,003 fact triplets.",
            "is_molecule_data": False
        },
        "FB15k_237": {
            "description": "The FB15K237 dataset from the 'Translating Embeddings for Modeling Multi-Relational Data' paper, containing 14,541 entities, 237 relations and 310,116 fact triples.",
            "is_molecule_data": False
        },
        "WikiCS": {
            "description": "The semi-supervised Wikipedia-based dataset from the 'Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks' paper, containing 11,701 nodes, 216,123 edges, 10 classes and 20 different training splits.",
            "is_molecule_data": False
        },
        "WebKB": {
            "description": "The WebKB datasets used in the 'Geom-GCN: Geometric Graph Convolutional Networks' paper.",
            "is_molecule_data": False
        },
        "WikipediaNetwork": {
            "description": "The Wikipedia networks introduced in the 'Multi-scale Attributed Node Embedding' paper.",
            "is_molecule_data": False
        },
        "HeterophilousGraphDataset": {
            "description": "The heterophilous graphs 'Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers' and 'Questions' from the 'A Critical Look at the Evaluation of GNNs under Heterophily: Are We Really Making Progress?' paper.",
            "is_molecule_data": False
        },
        "Actor": {
            "description": "The actor-only induced subgraph of the film-director-actor-writer network used in the 'Geom-GCN: Geometric Graph Convolutional Networks' paper.",
            "is_molecule_data": False
        },
        "UPFD": {
            "description": "The tree-structured fake news propagation graph classification dataset from the 'User Preference-aware Fake News Detection' paper.",
            "is_molecule_data": False
        },
        "GitHub": {
            "description": "The GitHub Web and ML Developers dataset introduced in the 'Multi-scale Attributed Node Embedding' paper.",
            "is_molecule_data": False
        },
        "FacebookPagePage": {
            "description": "The Facebook Page-Page network dataset introduced in the 'Multi-scale Attributed Node Embedding' paper.",
            "is_molecule_data": False
        },
        "LastFMAsia": {
            "description": "The LastFM Asia Network dataset introduced in the 'Characteristic Functions on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric Models' paper.",
            "is_molecule_data": False
        },
        "DeezerEurope": {
            "description": "The Deezer Europe dataset introduced in the 'Characteristic Functions on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric Models' paper.",
            "is_molecule_data": False
        },
        "GemsecDeezer": {
            "description": "The Deezer User Network datasets introduced in the 'GEMSEC: Graph Embedding with Self Clustering' paper.",
            "is_molecule_data": False
        },
        "Twitch": {
            "description": "The Twitch Gamer networks introduced in the 'Multi-scale Attributed Node Embedding' paper.",
            "is_molecule_data": False
        },
        "Airports": {
            "description": "The Airports dataset from the 'struc2vec: Learning Node Representations from Structural Identity' paper, where nodes denote airports and labels correspond to activity levels.",
            "is_molecule_data": False
        },
        "LRGBDataset": {
            "description": "The 'Long Range Graph Benchmark (LRGB)' datasets which is a collection of 5 graph learning datasets with tasks that are based on long-range dependencies in graphs.",
            "is_molecule_data": False
        },
        "MalNetTiny": {
            "description": "The MalNet Tiny dataset from the 'A Large-Scale Database for Graph Representation Learning' paper.",
            "is_molecule_data": False
        },
        "OMDB": {
            "description": "The Organic Materials Database (OMDB) of bulk organic crystals.",
            "is_molecule_data": False
        },
        "PolBlogs": {
            "description": "The Political Blogs dataset from the 'The Political Blogosphere and the 2004 US Election: Divided they Blog' paper.",
            "is_molecule_data": False
        },
        "EmailEUCore": {
            "description": "An e-mail communication network of a large European research institution, taken from the 'Local Higher-order Graph Clustering' paper.",
            "is_molecule_data": False
        },
        "LINKXDataset": {
            "description": "A variety of non-homophilous graph datasets from the 'Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods' paper.",
            "is_molecule_data": False
        },
        "EllipticBitcoinDataset": {
            "description": "The Elliptic Bitcoin dataset of Bitcoin transactions from the 'Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics' paper.",
            "is_molecule_data": False
        },
        "EllipticBitcoinTemporalDataset": {
            "description": "The time-step aware Elliptic Bitcoin dataset of Bitcoin transactions from the 'Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics' paper.",
            "is_molecule_data": False
        },
        "DGraphFin": {
            "description": "The DGraphFin networks from the 'DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection' paper.",
            "is_molecule_data": False
        },
        "HydroNet": {
            "description": "The HydroNet dataset from the 'HydroNet: Benchmark Tasks for Preserving Intermolecular Interactions and Structural Motifs in Predictive and Generative Models for Molecular Data' paper, consisting of 5 million water clusters held together by hydrogen bonding networks.",
            "is_molecule_data": True
        },
        "AirfRANS": {
            "description": "The AirfRANS dataset from the 'AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes Solutions' paper, consisting of 1,000 simulations of steady-state aerodynamics over 2D airfoils in a subsonic flight regime.",
            "is_molecule_data": False
        },
        "JODIEDataset": {
            "description": "The temporal graph datasets from the 'JODIE: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks' paper.",
            "is_molecule_data": False
        },
        "Wikidata5M": {
            "description": "The Wikidata-5M dataset from the 'KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation' paper, containing 4,594,485 entities, 822 relations, 20,614,279 train triples, 5,163 validation triples, and 5,133 test triples.",
            "is_molecule_data": False
        },
        "MyketDataset": {
            "description": "The Myket Android Application Install dataset from the 'Effect of Choosing Loss Function when Using T-Batching for Representation Learning on Dynamic Networks' paper.",
            "is_molecule_data": False
        },
        "BrcaTcga": {
            "description": "The breast cancer (BRCA TCGA Pan-Cancer Atlas) dataset consisting of patients with survival information and gene expression data from cBioPortal and a network of biological interactions between those nodes from Pathway Commons.",
            "is_molecule_data": False
        },
        "NeuroGraphDataset": {
            "description": "The NeuroGraph benchmark datasets from the 'NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics' paper.",
            "is_molecule_data": False
        }
    }
