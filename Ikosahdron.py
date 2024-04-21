import numpy as np
import plotly.graph_objs as go

class Icosahedron:
    def __init__(self):
        # Goldenes Verhältnis
        phi = (1 + np.sqrt(5)) / 2

        # Eckpunkte eines regulären Ikosaeders
        self.vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ])

        # Eckpunkte auf Einheitslänge skalieren
        self.vertices /= np.linalg.norm(self.vertices[0])

        # Flächen des Ikosaeders
        self.faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]

    def distance_to_point(self, point, x, y, z):
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

    def plot_distances_to_vector(self, distances):
        # Erstelle Mesh für Oberfläche mit den Distanzen zu dem Vektor
        vertices = self.vertices.tolist()
        faces = []
        for face in self.faces:
            faces.append([face[0], face[1], face[2]])

        mesh = go.Mesh3d(x=[v[0] for v in vertices], y=[v[1] for v in vertices], z=[v[2] for v in vertices],
                         i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
                         intensity=distances, colorscale='Viridis', flatshading=True)

        # Layout des Plots
        layout = go.Layout(
            title='Surface Distance from Nodes',
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
            )
        )

        # Plot anzeigen
        fig = go.Figure(data=[mesh], layout=layout)
        fig.show()


def plot_3d_graph(vertices, faces):
    # Create a list of vertex coordinates for each face
    face_vertices = [[vertices[vertex] for vertex in face] for face in faces]

    # Create a trace for each face
    traces = []
    for face in face_vertices:
        x, y, z = zip(*face)
        trace = go.Mesh3d(x=x, y=y, z=z, color='cyan', opacity=0.25)
        traces.append(trace)

    # Create layout
    layout = go.Layout(
        title='3D plot of a unit-length icosahedron',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='cube'
        )
    )

    # Plot
    fig = go.Figure(data=traces, layout=layout)
    fig.show()

if __name__ == '__main__':
    icosahedron = Icosahedron()
    # Annahme: Hier werden die Distanzen zu einem bestimmten Punkt berechnet
    # und dann zur Visualisierung übergeben
    # Zum Beispiel:
    distances = 0.5 * np.random.rand(len(icosahedron.vertices))  # Zufällige Distanzen
    icosahedron.plot_distances_to_vector(distances)

