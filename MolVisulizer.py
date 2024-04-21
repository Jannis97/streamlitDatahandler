from rdkit import Chem
from rdkit.Chem import AllChem
import plotly.graph_objs as go
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import use
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
use('TkAgg')
from skimage.morphology import local_maxima, local_minima
class MoleculeVisualizer:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.add_hydrogens_and_generate_3d_coordinates()

    def add_hydrogens_and_generate_3d_coordinates(self):
        # Füge Wasserstoffatome zum Molekül hinzu
        self.mol = Chem.AddHs(self.mol)
        # Generiere 3D-Koordinaten für das Molekül
        AllChem.EmbedMolecule(self.mol, AllChem.ETKDG())
        self.positions = self.mol.GetConformer().GetPositions()
        self.bonds = self.mol.GetBonds()
        self.symbols = [atom.GetSymbol() for atom in self.mol.GetAtoms()]

    def generate_dicts(self):
        # Erstelle ein Dictionary für jeden Knoten (Atom)
        atoms_dict = {}
        for i, pos in enumerate(self.positions):
            atoms_dict[i] = {
                'symbol': self.symbols[i],
                'position': pos,
                'index': i,
                'color': {'C': 'black', 'H': 'blue', 'N': 'green', 'O': 'red'}.get(self.symbols[i], 'gray'),
                'size': 30
            }

        bonds_dict = {}
        for i, bond in enumerate(self.bonds):
            start_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
            start_position = np.array(self.positions[start_atom_idx])
            end_position = np.array(self.positions[end_atom_idx])
            vector = end_position - start_position
            length = np.linalg.norm(vector)
            normalized_vector = vector / length if length != 0 else np.array([0,0,0])

            bonds_dict[(start_atom_idx, end_atom_idx)] = {
                'index': i,
                'start_atom_idx': start_atom_idx,
                'end_atom_idx': end_atom_idx,
                'start_position': start_position.tolist(),
                'end_position': end_position.tolist(),
                'color': 'gray',
                'width': 5,
                'length': length,
                'normalized_vector': normalized_vector.tolist()
            }

        return atoms_dict, bonds_dict

    def analyze_molecule(self):
        # Analyse der Hybridisierung der Atome und Bindungstypen
        atom_hybridization = {}
        bond_types = {}
        for atom in self.mol.GetAtoms():
            atom_hybridization[atom.GetIdx()] = {
                'symbol': atom.GetSymbol(),
                'hybridization': atom.GetHybridization().name
            }

        for bond in self.mol.GetBonds():
            bond_types[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = {
                'start_symbol': bond.GetBeginAtom().GetSymbol(),
                'end_symbol': bond.GetEndAtom().GetSymbol(),
                'bond_type': bond.GetBondType().name
            }

        return atom_hybridization, bond_types

    def combine_dicts(self, atoms_dict, atom_hybridization, bonds_dict, bond_types):
        # Aktualisiere die Dictionaries mit zusätzlichen Analyseinformationen
        for idx, atom in atom_hybridization.items():
            if idx in atoms_dict:
                atoms_dict[idx].update(atom)

        for idx, bond in bond_types.items():
            if idx in bonds_dict:
                bonds_dict[idx].update(bond)

        return atoms_dict, bonds_dict

    def calculate_bond_angles(self):
        # Erstelle ein Dictionary für die Atome und Bindungen
        atoms_dict, bonds_dict = self.generate_dicts()

        angles_mat = np.zeros((len(bonds_dict), len(bonds_dict)))
        # Gehe durch jedes Atom und berechne die Winkel zwischen den Bindungen
        for atom_idx in atoms_dict:
            connected_bonds = [bond_key for bond_key in bonds_dict if atom_idx in bond_key]

            # Wenn weniger als zwei Bindungen vorhanden sind, kann kein Winkel berechnet werden
            if len(connected_bonds) < 2:
                continue

            # Berechne die Winkel zwischen allen Bindungen, die am Atom ansetzen
            for i in range(len(connected_bonds)):
                for j in range(i + 1, len(connected_bonds)):
                    bond1_key = connected_bonds[i]
                    bond2_key = connected_bonds[j]
                    start0 = bond1_key[0]
                    end0 = bond1_key[1]
                    start1 = bond2_key[0]
                    end1 = bond2_key[1]
                    connectedAtoms = set([start0, end0, start1, end1])

                    if len(connectedAtoms) < 4:
                        bond0 = bonds_dict[bond1_key]
                        bond1 = bonds_dict[bond2_key]
                        vec1 = np.array(bond0['normalized_vector'])
                        vec2 = np.array(bond1['normalized_vector'])

                        # Berechne den Winkel zwischen den Vektoren
                        dot_product = np.dot(vec1, vec2)
                        angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi

                        idx0 = bond0['index']
                        idx1 = bond1['index']
                        angles_mat[idx0, idx1] = angle
                        angles_mat[idx1, idx0] = angle
        return angles_mat

    def distance_to_point(self,point, x, y, z):
        """
        Berechnet die Distanz zwischen einem Punkt und einem anderen Punkt.

        Args:
        point: Tuple oder Liste mit den Koordinaten des Punkts (x, y, z).
        x, y, z: Arrays mit den Koordinaten der Punkte, zu denen die Distanz berechnet werden soll.

        Returns:
        distances: Array mit den berechneten Distanzen.
        """
        # Extrahiere die Koordinaten des Punkts
        x_point, y_point, z_point = point

        # Berechne die Distanz zu jedem Punkt
        distances = np.sqrt((x - x_point) ** 2 + (y - y_point) ** 2 + (z - z_point) ** 2)

        return distances

    def plot_distances_to_vector(self, distances, x, y, z):

        # Erstellen eines 3D-Plots für die Distanzen zu dem Vektor
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot der Oberfläche mit den Distanzen zu dem Vektor
        surf = ax.plot_surface(x, y, z, facecolors=plt.cm.coolwarm(distances), alpha=0.8)

        # Einstellungen für den Plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Distances to {3}')

        # Hinzufügen einer Farbleiste (Colorbar)
        cbar = plt.colorbar(surf)
        cbar.set_label('Distance to vector')

        plt.show()

    def find_adjacent_bond(self):
        atoms_dict = self.atoms_dict
        bonds_dict = self.bonds_dict
        angle_matrix = self.calculate_bond_angles()
        for key in atoms_dict.keys():
            index = atoms_dict[key]['index']
            bondsCounter = 0
            adjBonds = {}
            for (begin, end) in bonds_dict.keys():
                if begin == index or end == index:
                    print(index, begin, end)
                    adjBonds[bondsCounter] = (begin, end)

                    bondsCounter += 1
            atoms_dict[key].update(adjBonds)

        for key, atom in atoms_dict.items():
            # Zähle die Anzahl der Bindungen für jedes Atom
            bond_count = sum(1 for (begin, end) in bonds_dict if begin == key or end == key)
            hybridization = atom['hybridization']

            symbol = atom['symbol']
            if hybridization == 'SP3' and bond_count == 2:
                # ether
                print(f"Atom {symbol}{key} mit SP3 Hybridisierung hat genau 2 Bindungen")
                a = atom[0]
                b = atom[1]

                idx0 = bonds_dict[a]['index']
                idx1 = bonds_dict[b]['index']
                angle = angle_matrix[idx0, idx1]
                print(f"Angle between bonds: {angle:.2f} degrees")
                atom_positions = np.array(atom['position'])
                vec0 = np.array(bonds_dict[a]['normalized_vector'])
                vec1 = np.array(bonds_dict[b]['normalized_vector'])

                vecHalf = -(vec0 + vec1) / 2

                # Berechnung des Bivektors als Kreuzprodukt
                bivector = np.cross(vec0, vec1)
                bivector /= np.linalg.norm(bivector)
                neg_bivector = -bivector
                print(f"Vector 0: {vec0}")
                print(f"Vector 1: {vec1}")
                print(f"Position: {atom_positions}")
                print(f"Angle: {angle}")
                print(f"Atom: {key}")
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x = np.outer(np.cos(u), np.sin(v))
                y = np.outer(np.sin(u), np.sin(v))
                z = np.outer(np.ones(np.size(u)), np.cos(v))

                distancetoVec0 = self.distance_to_point(vec0, x, y, z)
                #self.plot_distances_to_vector(distancetoVec0, x, y, z)

                distancetoVec1 = self.distance_to_point(vec1, x, y, z)
                #self.plot_distances_to_vector(distancetoVec1, x, y, z)

                distancetoNegBivector = self.distance_to_point(neg_bivector, x, y, z)
                #self.plot_distances_to_vector(distancetoNegBivector, x, y, z)

                distancetoBivector = self.distance_to_point(bivector, x, y, z)
                #self.plot_distances_to_vector(distancetoBivector, x, y, z)

                distancetoVecHalf = self.distance_to_point(vecHalf, x, y, z)
                #self.plot_distances_to_vector(distancetoVecHalf, x, y, z)

                stackedDist = np.stack((distancetoVec0, distancetoVec1, distancetoNegBivector, distancetoBivector, distancetoVecHalf))
                minDist = np.min(stackedDist, axis=0)
                #self.plot_distances_to_vector(minDist, x, y, z)

                # Find local minima
                minima_mask = local_minima(minDist, connectivity=1, allow_borders=True)

                # Create a plot to display both the data matrix and the local minima
                plt.figure(figsize=(12, 6))

                # Plotting the original data matrix
                plt.subplot(1, 2, 1)
                plt.title('Original Matrix')
                plt.imshow(minDist, cmap='viridis')
                plt.colorbar()

                # Plotting the local minima on the data matrix
                plt.subplot(1, 2, 2)
                plt.title('Local Minima Highlighted')
                # Overlay local minima on the original matrix for visualization
                # Minima are marked in red
                highlighted_minima = np.ma.masked_where(~minima_mask, minDist)
                plt.imshow(minDist, cmap='viridis')
                plt.imshow(highlighted_minima, cmap='autumn_r', interpolation='none')
                plt.colorbar()

                plt.show()
                a = 0
                '''
                print('norm of vec0:', np.linalg.norm(vec0))
                print('norm of vec1:', np.linalg.norm(vec1))

                dist_to_vec0 = np.linalg.norm(np.array([x, y, z]) - vec0.reshape(-1, 1, 1), axis=0)
                dist_to_vec1 = np.linalg.norm(np.array([x, y, z]) - vec1.reshape(-1, 1, 1), axis=0)
                dist_to_neg_bivector = np.linalg.norm(np.array([x, y, z]) - neg_bivector.reshape(-1, 1, 1), axis=0)
                
                # Bestimmung der maximalen Punkte basierend auf der Entfernung
                max_points_mask = np.logical_and(dist_to_vec0 == np.max(dist_to_vec0),
                                                 dist_to_vec1 == np.max(dist_to_vec1))

                # Färben der Oberfläche basierend auf der Nähe der Punkte
                min_dist = np.minimum(dist_to_vec0, dist_to_vec1)
                max_dist = np.maximum(dist_to_vec0, dist_to_vec1)
                color_array = (min_dist / max_dist)

                # Erstellen eines 3D-Plots
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # Plot der Oberfläche mit den entsprechenden Farben
                surf = ax.plot_surface(x, y, z, facecolors=plt.cm.coolwarm(color_array), alpha=0.8)

                # Markieren der maximalen Punkte auf der Oberfläche
                max_points = np.where(max_points_mask)
                ax.scatter(x[max_points], y[max_points], z[max_points], color='black', s=50, label='Maximal Points')

                # Erstellen und Hinzufügen einer Farbleiste (Colorbar)
                cbar = plt.colorbar(surf)
                cbar.set_label('Color based on proximity to vectors')

                # Einstellungen für den Plot
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('Vektoren und Atompositionen')
                ax.legend()

                plt.show()
                a = 0
                '''
    def calculate_bisector_vector(self, atom_index):
        # Finde die normierten Vektoren der Bindungen, die an diesem Atom ansetzen
        connected_bonds = [bond for bond in self.bonds if atom_index in bond.GetBeginAtomIdx() or atom_index in bond.GetEndAtomIdx()]

        if len(connected_bonds) != 2:
            raise ValueError("Die Berechnung der Winkelhalbierenden erfordert genau zwei Bindungen.")

        bond_vectors = []
        for bond in connected_bonds:
            # Bestimme den Vektor der Bindung
            start_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
            start_position = np.array(self.positions[start_atom_idx])
            end_position = np.array(self.positions[end_atom_idx])
            bond_vector = end_position - start_position
            normalized_vector = bond_vector / np.linalg.norm(bond_vector)
            bond_vectors.append(normalized_vector)

        # Addiere die beiden normierten Vektoren, um den Vektor der Winkelhalbierenden zu erhalten
        bisector_vector = np.add(bond_vectors[0], bond_vectors[1])
        bisector_vector_normalized = bisector_vector / np.linalg.norm(bisector_vector)

        return bisector_vector_normalized.tolist()

    def create_3d_plot(self):
        atoms_dict, bonds_dict = self.generate_dicts()
        atom_hybridization, bond_types = self.analyze_molecule()

        atoms_dict, bonds_dict = self.combine_dicts(atoms_dict, atom_hybridization, bonds_dict, bond_types)
        self.atoms_dict = atoms_dict
        self.bonds_dict = bonds_dict

        self.find_adjacent_bond()

        # Erstelle Traces für Atome und Bindungen
        atom_traces = [go.Scatter3d(
            x=[info['position'][0]],
            y=[info['position'][1]],
            z=[info['position'][2]],
            mode='markers+text',
            marker=dict(size=info['size'], color=info['color'], opacity=0.8),
            text=f"{info['symbol']} ({info['hybridization']}) {info['index']}",
            textposition='middle center',
            textfont=dict(size=16, color='black'),
            name=f"Atom {info['index'] + 1}"
        ) for info in atoms_dict.values()]

        bond_traces = [go.Scatter3d(
            x=[info['start_position'][0], info['end_position'][0], None],
            y=[info['start_position'][1], info['end_position'][1], None],
            z=[info['start_position'][2], info['end_position'][2], None],
            mode='lines',
            text=f"{info['bond_type']} (idx: {info['index']})",
            line=dict(color=info['color'], width=info['width']),
            name=f"{info['start_symbol']}-{info['end_symbol']} ({info['bond_type']})"
        ) for info in bonds_dict.values()]

        # Erstelle das Plotly-Figurenobjekt
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z')
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=True
        )
        fig = go.Figure(data=atom_traces + bond_traces, layout=layout)
        fig.show()

# Beispiel zur Verwendung der Klasse
smiles_string = "CO"  # Beispiel SMILES für Ethanol
visualizer = MoleculeVisualizer(smiles_string)
visualizer.create_3d_plot()
