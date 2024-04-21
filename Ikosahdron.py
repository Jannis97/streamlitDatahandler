import numpy as np
import plotly.graph_objs as go

class Icosahedron:
    def __init__(self):
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2

        # Vertices of a regular icosahedron
        self.vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ])

        # Scale vertices to unit length
        self.vertices /= np.linalg.norm(self.vertices[0])

        # Icosahedron faces
        self.faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        distancetoVec0 = self.distance_to_point(vec0, x, y, z)

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
    # Hier kommt der vorhandene Code hin
    def plot_distances_to_vector(self, distances, x, y, z):
        # Erstellen eines 3D-Plots für die Distanzen zu dem Vektor
        fig = go.Figure()

        # Trigonalpyramiden erstellen, um die Distanzen zu visualisieren
        for i in range(len(distances)):
            trace = go.Mesh3d(x=x[i], y=y[i], z=z[i], color='cyan', opacity=0.8)
            fig.add_trace(trace)

        # Einstellungen für den Plot
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
            ),
            title=f'Distances to Vector',
            margin=dict(l=0, r=0, t=40, b=0)
        )

        # Plot anzeigen
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
    plot_3d_graph(icosahedron.vertices, icosahedron.faces)
