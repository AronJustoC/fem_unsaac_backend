from .element import Element
from .node import Node


class Structure:
    """Representa una estructura compuesta por nodos y elementos.

    Permite añadir nodos, elementos y definir restricciones de grados de libertad.
    """

    def __init__(self):
        """Inicializa una nueva estructura vacía."""
        self.nodes = []
        self.elements = []
        self.num_dofs = 0
        self.constraints = {}  # Dictionary to store node constraints
        # Map DOF names to indices
        self.dof_map = {
            "ux": 0,
            "uy": 1,
            "uz": 2,  # translations
            "rx": 3,
            "ry": 4,
            "rz": 5,  # rotations
        }

    def add_node(self, node):
        """Añade un nuevo nodo a la estructura.

        Args:
            id (int): Id del nodo.
            x (float): Coordenada X del nodo.
            y (float): Coordenada Y del nodo.
            z (float): Coordenada Z del nodo.

        Returns:
            Node: El objeto Node recién creado.
        """
        # Asignar DOFs basados en el índice secuencial (posición en la lista)
        # Esto evita errores si los IDs no son secuenciales o son muy grandes
        node_idx = len(self.nodes)
        node.dofs = [node_idx * 6 + i for i in range(6)]
        self.nodes.append(node)
        self.num_dofs += 6  # 6 grados de libertad por nodo
        return node

    def add_element(self, element: Element):
        """Añade un nuevo elemento a la estructura.

        Args:
            element_id (int): Id del elemento.

        Returns:
            Element: El objeto Element añadido.
        """
        self.elements.append(element)
        return element

    def add_constraint(self, node: Node, constrained_dofs: list[str]):
        """Añade restricciones a un nodo específico de la estructura.

        Args:
            node (Node): El objeto Node al que se le aplicarán las restricciones.
            constrained_dofs (list[str]): Una lista de cadenas que representan
                                         los grados de libertad a restringir.
                                         Valores posibles: 'ux', 'uy', 'uz'
                                         (traslaciones) y 'rx', 'ry', 'rz'
                                         (rotaciones).
        Raises:
            ValueError: Si el nodo no forma parte de la estructura o si se
                        proporciona un nombre de DOF inválido.
        """
        if node not in self.nodes:
            raise ValueError("Node is not part of the structure")

        # Validate DOF names
        for dof in constrained_dofs:
            if dof not in self.dof_map:
                raise ValueError(f"Invalid DOF name: {dof}")

        # Store constraint indices for this node
        node_index = node.id
        constrained_indices = [self.dof_map[dof] for dof in constrained_dofs]
        self.constraints[node_index] = constrained_indices
        node.is_constrained = True

    def get_global_dof_index(self, node_id: int, local_dof: int) -> int:
        """Convierte un ID de nodo y un índice de DOF local a un índice de DOF global.

        Args:
            node_id (int): El ID del nodo.
            local_dof (int): El índice del grado de libertad local (0-5).

        Returns:
            int: El índice del grado de libertad global, o -1 si el nodo no existe.
        """
        node = next((n for n in self.nodes if n.id == node_id), None)
        if node:
            return node.dofs[local_dof]
        return -1

    def get_constrained_dofs(self) -> list[int]:
        """Obtiene una lista de todos los índices de grados de libertad globales restringidos.

        Returns:
            list[int]: Una lista ordenada de los índices de grados de libertad restringidos.
        """
        constrained_dofs = []
        for node_id, local_dofs in self.constraints.items():
            for local_dof in local_dofs:
                global_dof = self.get_global_dof_index(node_id, local_dof)
                if global_dof != -1:
                    constrained_dofs.append(global_dof)
        return sorted(list(set(constrained_dofs)))
