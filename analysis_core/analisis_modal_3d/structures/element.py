import numpy as np


class Element:
    """Representa un elemento estructural en 3D (viga).

    Calcula las matrices de rigidez y masa locales y globales para el elemento.
    """

    def __init__(self, element_id, node1, node2, section, material, consistent=False):
        """Inicializa un nuevo elemento.

        Args:
            element_id (int): Id del elemento.
            node1 (Node): El primer nodo del elemento.
            node2 (Node): El segundo nodo del elemento.
            section (dict): Un diccionario con las propiedades de la sección transversal
                            (e.g., 'area', 'Iy', 'Iz', 'Ix').
            material (dict): Un diccionario con las propiedades del material
                             (e.g., 'E', 'G', 'rho').
            consistent (bool): Si es True, usa matriz de masa consistente. Default False (lumped).
        """
        self.element_id = element_id
        self.nodes = (node1, node2)
        self.section = section
        self.material = material
        self.consistent = consistent
        self._compute_length()
        self._compute_local_stiffness()
        self._compute_transformation_matrix()
        self._compute_global_stiffness()
        self._compute_local_mass(consistent=self.consistent)
        self._compute_global_mass()

    def _compute_length(self):
        """Calcula la longitud del elemento y la almacena en self.L."""
        p1 = np.array(self.nodes[0].coords)
        p2 = np.array(self.nodes[1].coords)
        self.L = np.linalg.norm(p2 - p1)

    def _compute_local_stiffness(self):
        """Calcula la matriz de rigidez local del elemento y la almacena en self.k_local."""
        E = self.material["E"]
        G = self.material["G"]
        A = self.section["area"]
        Iy = self.section["Iy"]
        Iz = self.section["Iz"]
        J = self.section["J"]
        L = self.L

        # Parametros para la escritura dela matriz
        axial = E * A / L
        torsion = G * J / L

        # Terminos de flexion en el plano XY (flexion alrededor de Z)
        flex_xy_1 = 12 * E * Iz / L**3
        flex_xy_2 = 6 * E * Iz / L**2
        flex_xy_3 = 4 * E * Iz / L
        flex_xy_4 = 2 * E * Iz / L

        # Terminos de flexion en el plano XZ (flexion alrededor de Y)
        flex_xz_1 = 12 * E * Iy / L**3
        flex_xz_2 = 6 * E * Iy / L**2
        flex_xz_3 = 4 * E * Iy / L
        flex_xz_4 = 2 * E * Iy / L

        # Llenando la matriz de rigidez local (Simétrica)
        self.k_local = np.zeros((12, 12))
        
        # Nodo 1
        self.k_local[0,0] = self.k_local[6,6] = axial
        self.k_local[0,6] = self.k_local[6,0] = -axial
        
        self.k_local[3,3] = self.k_local[9,9] = torsion
        self.k_local[3,9] = self.k_local[9,3] = -torsion
        
        # Flexión XY (Alrededor de Z)
        self.k_local[1,1] = self.k_local[7,7] = flex_xy_1
        self.k_local[1,7] = self.k_local[7,1] = -flex_xy_1
        self.k_local[1,5] = self.k_local[5,1] = flex_xy_2
        self.k_local[1,11] = self.k_local[11,1] = flex_xy_2
        self.k_local[5,7] = self.k_local[7,5] = -flex_xy_2
        self.k_local[7,11] = self.k_local[11,7] = -flex_xy_2
        self.k_local[5,5] = self.k_local[11,11] = flex_xy_3
        self.k_local[5,11] = self.k_local[11,5] = flex_xy_4

        # Flexión XZ (Alrededor de Y) - Signos corregidos
        self.k_local[2,2] = self.k_local[8,8] = flex_xz_1
        self.k_local[2,8] = self.k_local[8,2] = -flex_xz_1
        self.k_local[2,4] = self.k_local[4,2] = -flex_xz_2
        self.k_local[2,10] = self.k_local[10,2] = -flex_xz_2
        self.k_local[4,8] = self.k_local[8,4] = flex_xz_2
        self.k_local[8,10] = self.k_local[10,8] = flex_xz_2
        self.k_local[4,4] = self.k_local[10,10] = flex_xz_3
        self.k_local[4,10] = self.k_local[10,4] = flex_xz_4

    def _compute_transformation_matrix(self):
        """Calcula la matriz de transformación del elemento (Coordenadas Globales Y-UP)."""
        p1 = np.array(self.nodes[0].coords)
        p2 = np.array(self.nodes[1].coords)
        dx = p2 - p1
        L = self.L

        if L < 1e-9:
            raise ValueError("La longitud del elemento no puede ser cero.")

        # Eje local x (dirección de la viga)
        lx, ly, lz = dx / L
        v_x = np.array([lx, ly, lz])

        # Caso especial: Viga vertical (paralela al eje Y global)
        # Usamos una pequeña tolerancia para detectar verticalidad
        if np.isclose(abs(ly), 1.0, atol=1e-6):
            # Para vigas verticales, definimos el eje local z como el eje Z global
            v_z = np.array([0, 0, 1])
            v_y = np.cross(v_z, v_x)
        else:
            # Para vigas no verticales, usamos el eje Y global [0,1,0] como referencia
            # El eje local z será horizontal (perpendicular al plano formado por v_x y Y)
            v_up = np.array([0, 1, 0])
            v_z = np.cross(v_x, v_up)
            v_z /= np.linalg.norm(v_z)
            # El eje local y completa el sistema ortonormal
            v_y = np.cross(v_z, v_x)

        # Matriz de rotación R (ejes locales como filas)
        R = np.vstack([v_x, v_y, v_z])

        self.T = np.zeros((12, 12))
        for i in range(4):
            self.T[i*3:(i+1)*3, i*3:(i+1)*3] = R

    def _compute_global_stiffness(self):
        """Calcula la matriz de rigidez global del elemento y la almacena en self.k_global."""
        self.k_global = self.T.T @ self.k_local @ self.T

    def get_stresses(self, u_global):
        """Calcula los esfuerzos de Von Mises en los dos extremos del elemento.
        u_global: Vector de desplazamientos globales (solo los 12 DOFs del elemento).
        """
        # Desplazamientos locales u_local = T @ u_global
        u_local = self.T @ u_global
        
        L = self.L
        E = self.material['E']
        G = self.material.get('G', E / (2 * (1 + self.material.get('nu', 0.3))))
        A = self.section['area']
        Iz = self.section['Iz']
        Iy = self.section['Iy']
        J = self.section['J']
        
        # Dimensiones para fibras extremas (usar valores por defecto si no existen)
        h = self.section.get('height', 0.1)
        b = self.section.get('width', 0.1)
        
        y_max = h / 2.0
        z_max = b / 2.0
        
        # Fuerzas internas locales (F = k_local @ u_local)
        f_local = self.k_local @ u_local
        
        stresses = []
        for i in [0, 6]: # Nodo 1 (i=0) y Nodo 2 (i=6)
            sgn = 1 if i == 0 else -1
            # Fuerzas en el extremo (convención de fuerza interna)
            Fx = -f_local[i] * sgn
            Fy = f_local[i+1] * sgn
            Fz = f_local[i+2] * sgn
            Mx = -f_local[i+3] * sgn
            My = f_local[i+4] * sgn
            Mz = f_local[i+5] * sgn
            
            # Esfuerzo normal (P/A + My*z/Iy + Mz*y/Iz)
            sigma_axial = Fx / A
            sigma_bend_y = abs(My * z_max / Iy)
            sigma_bend_z = abs(Mz * y_max / Iz)
            sigma_total = sigma_axial + sigma_bend_y + sigma_bend_z
            
            # Esfuerzo cortante (Torsión + Cortante directo con factor de forma para sección rectangular)
            kappa = 5.0 / 6.0
            tau_torsion = abs(Mx * max(y_max, z_max) / J)
            tau_vy = kappa * abs(Fy) / A
            tau_vz = kappa * abs(Fz) / A
            tau_total = tau_torsion + tau_vy + tau_vz
            
            # Von Mises: sqrt(sigma^2 + 3*tau^2)
            sigma_vm = np.sqrt(sigma_total**2 + 3 * tau_total**2)
            stresses.append(float(sigma_vm))
            
        return stresses

    def _compute_local_mass(self, consistent=None):
        """Calcula la matriz de masa local.
        consistent=True: Usa la matriz de masa consistente (más precisa para frecuencias).
        consistent=False: Usa masa concentrada (lumped).
        """
        if consistent is None:
            consistent = self.consistent
            
        rho = self.material["rho"]
        A = self.section["area"]
        L = self.L
        m = rho * A * L
        
        self.m_local = np.zeros((12, 12))
        
        if consistent:
            # Matriz de Masa Consistente (Standard for Euler-Bernoulli Beam)
            # Referencia: Przemieniecki, Theory of Matrix Structural Analysis
            
            # Axial (Indices 0, 6)
            axial_m = m / 6.0
            self.m_local[0,0] = self.m_local[6,6] = 2 * axial_m
            self.m_local[0,6] = self.m_local[6,0] = axial_m
            
            # Torsión (Indices 3, 9) - Asumiendo inercia polar J
            # Para vigas, Ip ≈ Iz + Iy
            Iz = self.section["Iz"]
            Iy = self.section["Iy"]
            Ip = Iz + Iy
            torsion_m = (rho * Ip * L) / 6.0
            self.m_local[3,3] = self.m_local[9,9] = 2 * torsion_m
            self.m_local[3,9] = self.m_local[9,3] = torsion_m
            
            # Flexión XY (Alrededor de Z) - (Indices 1, 5, 7, 11)
            c1 = m / 420.0
            self.m_local[1,1] = self.m_local[7,7] = 156 * c1
            self.m_local[1,7] = self.m_local[7,1] = 54 * c1
            self.m_local[1,5] = self.m_local[5,1] = 22 * L * c1
            self.m_local[1,11] = self.m_local[11,1] = -13 * L * c1
            self.m_local[5,7] = self.m_local[7,5] = 13 * L * c1
            self.m_local[7,11] = self.m_local[11,7] = -22 * L * c1
            self.m_local[5,5] = self.m_local[11,11] = 4 * L**2 * c1
            self.m_local[5,11] = self.m_local[11,5] = -3 * L**2 * c1
            
            # Flexión XZ (Alrededor de Y) - (Indices 2, 4, 8, 10)
            # Los signos pueden variar según la convención, pero la forma es similar a XY
            self.m_local[2,2] = self.m_local[8,8] = 156 * c1
            self.m_local[2,8] = self.m_local[8,2] = 54 * c1
            self.m_local[2,4] = self.m_local[4,2] = -22 * L * c1
            self.m_local[2,10] = self.m_local[10,2] = 13 * L * c1
            self.m_local[4,8] = self.m_local[8,4] = -13 * L * c1
            self.m_local[8,10] = self.m_local[10,8] = 22 * L * c1
            self.m_local[4,4] = self.m_local[10,10] = 4 * L**2 * c1
            self.m_local[4,10] = self.m_local[10,4] = -3 * L**2 * c1
            
        else:
            # Masa concentrada (Lumped)
            node_mass = m / 2.0
            
            # Inercias rotacionales lumped (aproximación física)
            Iz = self.section["Iz"]
            Iy = self.section["Iy"]
            Ip = Iz + Iy
            A = self.section["area"]
            L = self.L
            
            # Reducir la inercia rotacional para evitar acoplamiento excesivo y obtener modos "puros"
            # Un valor de 1.0 usa la inercia física. Valores muy pequeños (e.g. 1e-6) desacoplan
            # los grados de libertad rotacionales de la masa, lo cual es común en software comercial.
            rotary_factor = 1e-6
            
            # Rotación X (Torsión)
            jx = ((m * Ip) / (2.0 * A)) * rotary_factor
            # Rotación Y (Flexión XZ)
            jy = ((m * L**2) / 24.0 + (m * Iy) / (2.0 * A)) * rotary_factor
            # Rotación Z (Flexión XY)
            jz = ((m * L**2) / 24.0 + (m * Iz) / (2.0 * A)) * rotary_factor

            # Llenar diagonal de la matriz de masa local
            # Nodo 1 (0-5)
            self.m_local[0,0] = self.m_local[1,1] = self.m_local[2,2] = node_mass
            self.m_local[3,3] = jx
            self.m_local[4,4] = jy
            self.m_local[5,5] = jz
            
            # Nodo 2 (6-11)
            self.m_local[6,6] = self.m_local[7,7] = self.m_local[8,8] = node_mass
            self.m_local[9,9] = jx
            self.m_local[10,10] = jy
            self.m_local[11,11] = jz


    def _compute_global_mass(self):
        """Transforma la matriz de masa local a coordenadas globales."""
        self.m_global = self.T.T @ self.m_local @ self.T

    @property
    def dof_indices(self):
        """Retorna una lista con los índices de los 12 grados de libertad del elemento."""
        return self.nodes[0].dofs + self.nodes[1].dofs

    def get_internal_forces(self, global_displacements):
        """
        Calcula las fuerzas internas en el elemento a partir de los desplazamientos globales.
        """
        dofs_node1 = self.nodes[0].dofs
        dofs_node2 = self.nodes[1].dofs

        element_global_displacements = np.zeros(12)
        element_global_displacements[0:6] = global_displacements[dofs_node1]
        element_global_displacements[6:12] = global_displacements[dofs_node2]

        element_local_displacements = self.T @ element_global_displacements
        local_forces = self.k_local @ element_local_displacements

        return {
            "fx1": local_forces[0], "fy1": local_forces[1], "fz1": local_forces[2],
            "mx1": local_forces[3], "my1": local_forces[4], "mz1": local_forces[5],
            "fx2": local_forces[6], "fy2": local_forces[7], "fz2": local_forces[8],
            "mx2": local_forces[9], "my2": local_forces[10], "mz2": local_forces[11],
        }
