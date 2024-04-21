# For plotting the normal vectors to the faces of the icosahedron
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create a sphere
phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(phi), np.cos(theta))
y = np.outer(np.sin(phi), np.sin(theta))
z = np.outer(np.cos(phi), np.ones_like(theta))

# Icosahedron vertices
t = (1.0 + np.sqrt(5.0)) / 2.0
icosahedron_vertices = np.array([
    [-1,  t,  0], [1,  t,  0], [-1, -t,  0], [1, -t,  0],
    [0, -1,  t], [0,  1,  t], [0, -1, -t], [0,  1, -t],
    [t,  0, -1], [t,  0,  1], [-t, 0, -1], [-t, 0,  1]
])

# Normalize the vertices so that they lie on the unit sphere
icosahedron_vertices /= np.linalg.norm(icosahedron_vertices, axis=1)[:, np.newaxis]

# Icosahedron faces
faces = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
]

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='b', alpha=0.1)

# Draw vertices
ax.scatter(icosahedron_vertices[:, 0], icosahedron_vertices[:, 1], icosahedron_vertices[:, 2], color='r')

# Draw edges
for face in faces:
    for i in range(3):
        start_vertex = icosahedron_vertices[face[i]]
        end_vertex = icosahedron_vertices[face[(i + 1) % 3]]
        ax.plot([start_vertex[0], end_vertex[0]], [start_vertex[1], end_vertex[1]], [start_vertex[2], end_vertex[2]], color='r')

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_aspect("auto")
ax.set_title("Icosahedron on Unit Sphere")
plt.show()

# Function to calculate the normal of a triangle given its vertices
def triangle_normal(v1, v2, v3):
    return np.cross(v2 - v1, v3 - v1)

# New plot with normals
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='b', alpha=0.1)

# Plot vertices and edges as before
ax.scatter(icosahedron_vertices[:, 0], icosahedron_vertices[:, 1], icosahedron_vertices[:, 2], color='r')
for face in faces:
    for i in range(3):
        start_vertex = icosahedron_vertices[face[i]]
        end_vertex = icosahedron_vertices[face[(i + 1) % 3]]
        ax.plot([start_vertex[0], end_vertex[0]], [start_vertex[1], end_vertex[1]], [start_vertex[2], end_vertex[2]], color='r')

# Calculate and plot normal vectors for each face
normals = np.array([triangle_normal(icosahedron_vertices[face[0]], icosahedron_vertices[face[1]], icosahedron_vertices[face[2]])
                    for face in faces])
normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]  # Normalize the normals

midpoints = np.array([np.mean(icosahedron_vertices[face], axis=0) for face in faces])  # Compute midpoints of each face for vector origin

# Draw normal vectors
for midpoint, normal in zip(midpoints, normals):
    ax.quiver(midpoint[0], midpoint[1], midpoint[2], normal[0], normal[1], normal[2], length=0.1, color='g', linewidth=1.5)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_aspect("auto")
ax.set_title("Icosahedron with Normal Vectors on Unit Sphere")
plt.show()

# Convert integer vertices to float to avoid type errors during normalization
tetrahedron_vertices = tetrahedron_vertices.astype(float)
cube_vertices = cube_vertices.astype(float)
octahedron_vertices = octahedron_vertices.astype(float)
dodecahedron_vertices = dodecahedron_vertices.astype(float)

# Normalizing vertices for each Platonic solid
tetrahedron_vertices /= np.linalg.norm(tetrahedron_vertices, axis=1)[:, np.newaxis]
cube_vertices /= np.linalg.norm(cube_vertices, axis=1)[:, np.newaxis]
octahedron_vertices /= np.linalg.norm(octahedron_vertices, axis=1)[:, np.newaxis]
dodecahedron_vertices /= np.linalg.norm(dodecahedron_vertices, axis=1)[:, np.newaxis]

# Create plots for each Platonic solid
fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(12, 12))

# Function to plot Platonic solids
def plot_solid(ax, vertices, faces, title):
    ax.plot_surface(x, y, z, color='b', alpha=0.1)
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='r')
    for face in faces:
        for i in range(len(face)):
            start_vertex = vertices[face[i]]
            end_vertex = vertices[face[(i + 1) % len(face)]]
            ax.plot([start_vertex[0], end_vertex[0]], [start_vertex[1], end_vertex[1]], [start_vertex[2], end_vertex[2]], color='r')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("auto")
    ax.set_title(title)

# Plot each Platonic solid
plot_solid(axs[0, 0], tetrahedron_vertices, tetrahedron_faces, "Tetrahedron")
plot_solid(axs[0, 1], cube_vertices, cube_faces, "Cube (Hexahedron)")
plot_solid(axs[1, 0], octahedron_vertices, octahedron_faces, "Octahedron")
plot_solid(axs[1, 1], dodecahedron_vertices, dodecahedron_faces, "Dodecahedron")

plt.tight_layout()
plt.show()


import numpy as np
from scipy.optimize import minimize

def distance(p1, p2):
    return np.arccos(np.dot(p1, p2))

def objective_function(x):
    n = len(x) // 3
    points = np.reshape(x, (n, 3))
    min_distance = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            dist = distance(points[i], points[j])
            if dist < min_distance:
                min_distance = dist
    return -min_distance  # We maximize, so negate the distance

def constraint_function(x):
    return np.sum(x**2) - len(x)  # Constraint to keep points on the unit sphere

n = 10  # Number of points
initial_guess = np.random.randn(n * 3)  # Initial guess for optimization

# Optimization
result = minimize(objective_function, initial_guess, constraints={'type': 'eq', 'fun': constraint_function})
optimized_points = np.reshape(result.x, (n, 3))

print("Optimized points on the unit sphere:")
print(optimized_points)
