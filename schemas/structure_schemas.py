from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


# Modelo para el nodo
class Node(BaseModel):
    id: int = Field(..., description="Identificador unico del nodo")
    coords: tuple[float, float,
                  float] = Field(..., description="Coordenadas (x,y,z) del nodo")
    mass: float = Field(0.0, description="Masa puntual en el nodo (kg)")
    model_config = ConfigDict(extra='ignore')


# Modelo para le material
class Material(BaseModel):
    id: int = Field(..., description="Identificador unico del material")
    name: Optional[str] = Field(None, description="Nombre del material")
    E: float = Field(..., description="Modulo de Young(elasticidad)")
    nu: float = Field(..., description="Coeficiente de Poisson")
    rho: float = Field(..., description="Densidad del material")
    G: Optional[float] = Field(None, description="Modulo de cortante")
    yield_strength: float = Field(250e6, description="Esfuerzo de fluencia (Pa)")
    
    model_config = ConfigDict(extra='ignore')

    # Otras propiedades...


# Modelo para la seccion
class Section(BaseModel):
    id: int = Field(..., description="Identificador unico de seccion")
    name: Optional[str] = Field(None, description="Nombre de la seccion")
    # Acepta tanto 'area' como 'A' (alias usado en frontend antiguo)
    area: float = Field(..., gt=0, description="Area de ka seccion transversal", alias="A")
    Iz: float = Field(..., gt=0,
                      description="Momento de inercia alrededor de eje local z")
    Iy: float = Field(..., gt=0,
                      description="Momento de inercia alrededor de eje local y")
    J: float = Field(..., gt=0,
                     description="Constante torsional")
    
    model_config = ConfigDict(populate_by_name=True, extra='ignore')


# Modelo para Elemento
class Element(BaseModel):
    id: int = Field(..., description="Identificador unico de elemento")
    node_ids: tuple[int, int] = Field(...,
                                      description="Ids de los nodos que conecta")
    material_id: int = Field(...,
                             description="ID del material asignada al elemento")
    section_id: int = Field(...,
                            description="ID de la seccion asignada al elemento")
    
    model_config = ConfigDict(extra='ignore')


# Modelo para cargas en nodos
class NodalLoad(BaseModel):
    id: Optional[int] = Field(None, description="ID de la carga (opcional)")
    name: Optional[str] = Field(None, description="Nombre de la carga (opcional)")
    node_id: int = Field(...,
                         description="ID del nodo donde se aplica la carga")
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0
    
    model_config = ConfigDict(extra='ignore')


# Modelo principal para ingresar la estructura
class StructureInput(BaseModel):
    nodes: list[Node] = Field(...,
                              description="lista de nodos de la estructura")
    materials: list[Material] = Field(...,
                                      description="lista de materiales definidos")
    sections: list[Section] = Field(...,
                                    description="lista de secciones definidas")
    elements: list[Element] = Field(...,
                                    description="lista de elementos de la estructura")
    loads: list[NodalLoad] = Field(
        default_factory=list, description="lista de cargas aplicadas")
    restraints: dict[int, list[str]] = Field(
        default_factory=dict, description="Restricciones del GDL por nodo ({0:[\"ux\", \"uy\", \"uz\"})")


# Modelos para resultados

class StaticResults(BaseModel):
    displacements: dict[int, list[float]] = Field(
        ..., description="Desplazamientos por nodo(ID: [ux,uy,uz,rx,ry,rz])")
    element_forces: dict[int, dict[str, float]] = Field(..., description="Fuerzas internas en los elementos")
    reactions: dict[int, list[float]] = Field(
        ..., description="Reacciones en los apoyos por nodo (ID: [fx,fy,fz,mx,my,mz])")
    stresses: dict[int, list[float]] = Field(default_factory=dict, description="Esfuerzos de Von Mises por elemento (ID: [sigma_node1, sigma_node2])")


class ModalResults(BaseModel):
    frequencies: list[float] = Field(...,
                                     description="Frecuencias naturales de vibracion")
    mode_shapes: dict[int, list[list[float]]] = Field(
        ..., description="Modos de vibracion por nodo (ID: [[ux1,uy2,...],[ux2,uy2,...])")
    mass_participation: Optional[list[list[float]]] = Field(None, description="Porcentaje de participación de masa modal [X, Y, Z] por modo")


class HarmonicResults(BaseModel):
    frequencies_sweep: list[float] = Field(...,
                                           description="Rango de frecuencias analizadas")
    # node_id -> [amplitudes para cada frecuencia]
    response_amplitudes: dict[int, list[float]] = Field(
        ..., description="Amplitudes de respuesta por nodo para cada frecuencia")


class HarmonicAnalysisRequest(BaseModel):
    structure: StructureInput
    freq_start: float = Field(0.1, description="Frecuencia inicial del barrido (Hz)")
    freq_end: float = Field(100.0, description="Frecuencia final del barrido (Hz)")
    num_points: int = Field(100, description="Numero de puntos en el barrido")
    damping_ratio: float = Field(0.05, description="Relación de amortiguamiento (ej, 0.05 para 5%)")
    is_unbalanced: bool = Field(False, description="Si es una fuerza de desbalance proporcional a w^2")
    unbalanced_me: float = Field(0.0, description="Producto masa x excentricidad (m*e)")
