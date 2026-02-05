class Node:
    """Representa un nodo en una estructura 3D.

    Cada nodo tiene un ID único, coordenadas (x, y, z) y grados de libertad (DOFs).
    La masa puntual también se puede asignar a un nodo.
    """

    def __init__(self, id, x, y, z):
        """Inicializa un nuevo nodo.

        Args:
            node_id (int): Id del nodo
            x (float): Coordenada X del nodo.
            y (float): Coordenada Y del nodo.
            z (float): Coordenada Z del nodo.
        """
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.coords = (x, y, z)
        self.dofs = []  # DOFs serán asignados por la clase Structure
        self.mass = 0.0  # Masa puntual inicializada en 0
        self.is_constrained = False

    def __repr__(self):
        return f"Node {self.id} ({self.coords})"
