from fastapi import FastAPI, HTTPException, status, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import json
from collections import OrderedDict
from threading import Lock
from typing import Literal

from schemas.structure_schemas import StructureInput, StaticResults, ModalResults, HarmonicAnalysisRequest, HarmonicResults

CACHE_MAX_SIZE = 100


class LRUCache:
    """Thread-safe LRU cache with size limit. Uses OrderedDict for O(1) access and eviction."""
    
    def __init__(self, max_size: int = CACHE_MAX_SIZE):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None
    
    def set(self, key: str, value):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
    
    def stats(self):
        with self._lock:
            return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}


_static_cache = LRUCache(CACHE_MAX_SIZE)
_modal_cache = LRUCache(CACHE_MAX_SIZE)
_modal_viz_cache = LRUCache(50)
_harmonic_cache = LRUCache(CACHE_MAX_SIZE)
_element_matrix_cache = LRUCache(CACHE_MAX_SIZE)
_element_bd_cache = LRUCache(CACHE_MAX_SIZE)
_global_matrix_bundle_cache = LRUCache(20)
_impedance_matrix_bundle_cache = LRUCache(20)


def _sort_nested(obj):
    if isinstance(obj, dict):
        return {k: _sort_nested(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [_sort_nested(item) for item in obj]
    return obj


def compute_structure_hash(structure_data: dict, extra_params: dict | None = None) -> str:
    """Stable hash via recursive key sorting before JSON serialization."""
    hash_input: dict = {"structure": _sort_nested(structure_data)}
    if extra_params:
        hash_input["params"] = _sort_nested(extra_params)
    
    json_str = json.dumps(hash_input, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
from analysis_core.api_adapters import (
    build_global_matrix_bundle,
    build_impedance_matrix_bundle,
    get_element_matrices,
    get_element_bd_matrices,
    get_global_matrix_view,
    get_impedance_matrix_view,
    run_static_analysis,
    run_modal_analysis,
    run_harmonic_analysis,
)
from analysis_core.analisis_modal_3d.data.engineering_library import get_library
from pydantic import BaseModel, Field
from fastapi import Depends
from supabase_client import get_current_user, supabase
from tasks import run_analysis_task
from storage import download_result
from signal_core.signal_api import router as signal_router

app = FastAPI(title="API de Análisis Estructural y Procesamiento de Señales",
              description="API para realizar análisis estáticos, modales, armónicos de estructuras y procesamiento de señales de vibración para puentes",
              version="1.0.0")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "fem-backend"}

# ... (rest of middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar rutas de procesamiento de señales
app.include_router(signal_router)

@app.get("/api/library/standard", summary="Obtiene la biblioteca estandar de materiales y secciones")
async def get_standard_library():
    """
    Retorna un diccionario con materiales y secciones estándar (Aceros, Aluminios, Perfiles IPE, HEB, etc).
    """
    return get_library()

@app.post("/api/analysis/static", response_model=StaticResults, summary="Realiza el analisis estatico de la estructura")
async def perform_static_analysis(structure_input: StructureInput):
    try:
        structure_data = structure_input.model_dump()
        cache_key = compute_structure_hash(structure_data, {"analysis": "static"})
        
        cached = _static_cache.get(cache_key)
        if cached is not None:
            print(f"[CACHE HIT] Static analysis - key: {cache_key[:16]}...")
            return StaticResults(**cached)
        
        print(f"\n{'='*60}")
        print(f"NUEVA PETICION DE ANALISIS ESTATICO")
        print(f"{'='*60}")
        print(f"Datos recibidos:")
        print(f"  - Nodos: {len(structure_input.nodes)}")
        print(f"  - Elementos: {len(structure_input.elements)}")
        print(f"  - Materiales: {len(structure_input.materials)}")
        print(f"  - Secciones: {len(structure_input.sections)}")
        print(f"  - Cargas: {len(structure_input.loads)}")
        print(f"  - Restricciones: {len(structure_input.restraints)}")
        print(f"{'='*60}\n")
        
        result_dict = run_static_analysis(structure_data)
        if "error" in result_dict:
            print(f"ERROR EN ANALISIS: {result_dict['error']}")
            raise HTTPException(status_code=400, detail=result_dict["error"])
        
        _static_cache.set(cache_key, result_dict)
        print(f"[CACHE SET] Static analysis - key: {cache_key[:16]}...")
        
        print("DEBUG: Validando resultados con Pydantic...")
        try:
            validated_results = StaticResults(**result_dict)
            print("DEBUG: Validación exitosa")
            return validated_results
        except Exception as validation_error:
            print(f"ERROR DE VALIDACION PYDANTIC:")
            print(f"  Tipo: {type(validation_error).__name__}")
            print(f"  Mensaje: {str(validation_error)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error validando resultados: {str(validation_error)}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"EXCEPTION EN ANALISIS: {str(e)}")
        print(f"Tipo de excepción: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error inesperado en el analisis estatico: {str(e)}")


class ModalAnalysisRequest(BaseModel):
    structure: StructureInput
    num_modes: int


class ElementMatricesRequest(BaseModel):
    structure: StructureInput
    element_id: int


class GlobalMatrixViewRequest(BaseModel):
    structure: StructureInput
    matrix_kind: Literal['stiffness', 'mass', 'damping'] = 'stiffness'
    matrix_scope: Literal['full', 'free'] = 'full'
    damping_ratio: float = Field(0.02, ge=0, description="Solo aplica si matrix_kind='damping' (Rayleigh, calibrado con la 1ra frecuencia natural)")
    row_start: int = Field(0, ge=0)
    col_start: int = Field(0, ge=0)
    window_size: int = Field(12, ge=6, le=36)
    heatmap_bins: int = Field(64, ge=16, le=120)

class ImpedanceMatrixViewRequest(BaseModel):
    structure: StructureInput
    frequency_hz: float = Field(..., ge=0)
    damping_ratio: float = Field(0.05, ge=0)
    matrix_scope: Literal['full', 'free'] = 'free'
    row_start: int = Field(0, ge=0)
    col_start: int = Field(0, ge=0)
    window_size: int = Field(12, ge=6, le=36)
    heatmap_bins: int = Field(64, ge=16, le=120)

class AnalysisRequest(BaseModel):
    project_id: str
    analysis_type: str
    num_modes: int = 12

@app.post("/api/analysis/calculate")
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    proj_res = supabase.table("projects").select("structure_data").eq("id", request.project_id).eq("user_id", user["sub"]).execute()
    if not proj_res.data:
        raise HTTPException(status_code=404, detail="Project not found")
    
    structure_data = proj_res.data[0]["structure_data"]
    
    analysis_res = supabase.table("analyses").insert({
        "project_id": request.project_id,
        "user_id": user["sub"],
        "analysis_type": request.analysis_type,
        "status": "pending",
        "progress": 0
    }).execute()
    
    analysis_id = analysis_res.data[0]["id"]
    
    background_tasks.add_task(
        run_analysis_task,
        analysis_id=analysis_id,
        structure_data=structure_data,
        analysis_type=request.analysis_type,
        num_modes=request.num_modes
    )
    
    return {"analysis_id": analysis_id, "status": "pending"}

@app.get("/api/analysis/{analysis_id}")
async def get_analysis_status(analysis_id: str, user=Depends(get_current_user)):
    res = supabase.table("analyses").select("*").eq("id", analysis_id).eq("user_id", user["sub"]).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return res.data[0]

@app.get("/api/analysis/{analysis_id}/result")
async def get_analysis_result(analysis_id: str, user=Depends(get_current_user)):
    analysis = supabase.table("analyses").select("*").eq("id", analysis_id).eq("user_id", user["sub"]).execute()
    if not analysis.data:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if analysis.data[0]["status"] != "done":
        raise HTTPException(status_code=400, detail="Analysis not complete")
    
    result = download_result(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return result

@app.post("/api/analysis/modal", response_model=ModalResults, summary="Realiza el analisis modal de la estructura")
async def perform_modal_analysis(request: ModalAnalysisRequest):
    try:
        structure_data = request.structure.model_dump()
        cache_key = compute_structure_hash(structure_data, {"analysis": "modal", "num_modes": request.num_modes})
        
        cached = _modal_cache.get(cache_key)
        if cached is not None:
            print(f"[CACHE HIT] Modal analysis - key: {cache_key[:16]}...")
            return ModalResults(**cached)
        
        result_dict = run_modal_analysis(structure_data, request.num_modes)
        if "error" in result_dict:
            raise HTTPException(status_code=400, detail=result_dict["error"])
        
        _modal_cache.set(cache_key, result_dict)
        print(f"[CACHE SET] Modal analysis - key: {cache_key[:16]}...")
        
        return ModalResults(**result_dict)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error inesperado en el analisis modal: {str(e)}")


@app.post("/api/analysis/element-matrices", summary="Obtiene matrices FEM de un elemento")
async def element_matrices(request: ElementMatricesRequest):
    structure_data = request.structure.model_dump()
    cache_key = compute_structure_hash(
        structure_data,
        {"analysis": "element-matrices", "element_id": request.element_id},
    )
    cached = _element_matrix_cache.get(cache_key)
    if cached is not None:
        return cached

    result = get_element_matrices(structure_data, request.element_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    _element_matrix_cache.set(cache_key, result)
    return result


@app.post("/api/analysis/element-bd-matrices", summary="Matrices [B] y [D] explicitas de un elemento, con verificacion contra k_local")
async def element_bd_matrices(request: ElementMatricesRequest):
    structure_data = request.structure.model_dump()
    cache_key = compute_structure_hash(
        structure_data,
        {"analysis": "element-bd-matrices", "element_id": request.element_id},
    )
    cached = _element_bd_cache.get(cache_key)
    if cached is not None:
        return cached

    result = get_element_bd_matrices(structure_data, request.element_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    _element_bd_cache.set(cache_key, result)
    return result


@app.post("/api/analysis/global-matrices", summary="Explora matrices globales ensambladas")
async def global_matrices(request: GlobalMatrixViewRequest):
    """Devuelve un mapa completo y una ventana numérica de K/M o Kff/Mff."""
    structure_data = request.structure.model_dump()
    bundle_key = compute_structure_hash(
        structure_data,
        {"analysis": "global-matrix-bundle"},
    )
    bundle = _global_matrix_bundle_cache.get(bundle_key)
    if bundle is None:
        try:
            bundle = build_global_matrix_bundle(structure_data)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"No se pudo ensamblar K/M: {exc}") from exc
        _global_matrix_bundle_cache.set(bundle_key, bundle)

    result = get_global_matrix_view(
        bundle,
        matrix_kind=request.matrix_kind,
        matrix_scope=request.matrix_scope,
        damping_ratio=request.damping_ratio,
        row_start=request.row_start,
        col_start=request.col_start,
        window_size=request.window_size,
        heatmap_bins=request.heatmap_bins,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/analysis/impedance-matrix", summary="Explora la matriz de impedancia dinamica Z(w) = K - w^2 M + i w C")
async def impedance_matrix(request: ImpedanceMatrixViewRequest):
    """Devuelve un mapa completo y una ventana numérica (Re/Im) de Z(ω) para una frecuencia dada."""
    structure_data = request.structure.model_dump()

    bundle_key = compute_structure_hash(structure_data, {"analysis": "global-matrix-bundle"})
    km_bundle = _global_matrix_bundle_cache.get(bundle_key)
    if km_bundle is None:
        try:
            km_bundle = build_global_matrix_bundle(structure_data)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"No se pudo ensamblar K/M: {exc}") from exc
        _global_matrix_bundle_cache.set(bundle_key, km_bundle)

    impedance_key = compute_structure_hash(
        structure_data,
        {"analysis": "impedance-matrix-bundle", "damping_ratio": request.damping_ratio},
    )
    impedance_bundle = _impedance_matrix_bundle_cache.get(impedance_key)
    if impedance_bundle is None:
        try:
            impedance_bundle = build_impedance_matrix_bundle(km_bundle, damping_ratio=request.damping_ratio)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"No se pudo ensamblar la matriz de impedancia: {exc}") from exc
        _impedance_matrix_bundle_cache.set(impedance_key, impedance_bundle)

    result = get_impedance_matrix_view(
        impedance_bundle,
        frequency_hz=request.frequency_hz,
        matrix_scope=request.matrix_scope,
        row_start=request.row_start,
        col_start=request.col_start,
        window_size=request.window_size,
        heatmap_bins=request.heatmap_bins,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/analysis/harmonic", response_model=HarmonicResults, summary="Realiza el analisis armonico (respuesta en frecuencia)")
async def perform_harmonic_analysis(request: HarmonicAnalysisRequest):
    try:
        request_data = request.model_dump()
        cache_key = compute_structure_hash(
            request.structure.model_dump(),
            {
                "analysis": "harmonic",
                "freq_start": request.freq_start,
                "freq_end": request.freq_end,
                "num_points": request.num_points,
                "damping_ratio": request.damping_ratio,
                "is_unbalanced": request.is_unbalanced,
                "unbalanced_me": request.unbalanced_me,
                "unbalanced_node_id": request.unbalanced_node_id,
                "unbalanced_direction": request.unbalanced_direction,
                "unbalanced_plane": request.unbalanced_plane,
                "unbalanced_mass": request.unbalanced_mass,
                "unbalanced_eccentricity": request.unbalanced_eccentricity,
            },
        )

        cached = _harmonic_cache.get(cache_key)
        if cached is not None:
            print(f"[CACHE HIT] Harmonic analysis - key: {cache_key[:16]}...")
            return HarmonicResults(**cached)

        result_dict = run_harmonic_analysis(request_data)
        if "error" in result_dict:
            raise HTTPException(status_code=400, detail=result_dict["error"])

        _harmonic_cache.set(cache_key, result_dict)
        print(f"[CACHE SET] Harmonic analysis - key: {cache_key[:16]}...")
        return HarmonicResults(**result_dict)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error inesperado en el analisis armonico: {str(e)}")


from visualization.plotly_engine import generate_structure_figure, generate_results_figure
import json

@app.post("/api/visualization/structure")
async def get_structure_viz(structure_input: StructureInput, theme: str = "dark"):
    fig = generate_structure_figure(structure_input.model_dump(), theme=theme)
    return Response(content=fig.to_json(), media_type="application/json")


@app.post("/api/visualization/static-results")
async def get_static_results_viz(structure_input: StructureInput, scale: float = 1.0, theme: str = "dark"):
    try:
        structure_data = structure_input.model_dump()
        cache_key = compute_structure_hash(structure_data, {"analysis": "static"})
        
        cached = _static_cache.get(cache_key)
        if cached is not None:
            print(f"[CACHE HIT] Static viz - key: {cache_key[:16]}...")
            result_dict = cached
        else:
            result_dict = run_static_analysis(structure_data)
            if "error" in result_dict:
                raise HTTPException(status_code=400, detail=result_dict["error"])
            _static_cache.set(cache_key, result_dict)
            print(f"[CACHE SET] Static viz - key: {cache_key[:16]}...")

        fig = generate_results_figure(
            structure_data, result_dict["displacements"], scale, theme=theme)
        return Response(content=fig.to_json(), media_type="application/json")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/visualization/modal-results")
async def get_modal_results_viz(
    structure_input: StructureInput,
    mode_index: int = 0,
    num_modes: int = 12,
    scale: float = 1.0,
    theme: str = "dark",
    animate: bool = False,
    frame_count: int = 10,
    detail: int = 2,
):
    try:
        structure_data = structure_input.model_dump()
        cache_key = compute_structure_hash(structure_data, {"analysis": "modal", "num_modes": num_modes})
        
        cached = _modal_cache.get(cache_key)
        if cached is not None:
            print(f"[CACHE HIT] Modal viz - key: {cache_key[:16]}...")
            result_dict = cached
        else:
            result_dict = run_modal_analysis(structure_data, num_modes)
            if "error" in result_dict:
                raise HTTPException(status_code=400, detail=result_dict["error"])
            _modal_cache.set(cache_key, result_dict)
            print(f"[CACHE SET] Modal viz - key: {cache_key[:16]}...")

        mode_shapes = result_dict.get("mode_shapes", {})
        frequencies = result_dict.get("frequencies", [])

        if not frequencies:
            raise HTTPException(
                status_code=400, detail="No se encontraron modos de vibración")

        if mode_index >= len(frequencies):
            mode_index = 0

        detail = max(1, min(int(detail), 6))
        frame_count = max(4, min(int(frame_count), 30))
        viz_cache_key = compute_structure_hash(
            structure_data,
            {
                "analysis": "modal-viz",
                "num_modes": num_modes,
                "mode_index": mode_index,
                "scale": round(float(scale), 8),
                "theme": theme,
                "animate": animate,
                "frame_count": frame_count if animate else 0,
                "detail": detail,
            },
        )
        cached_json = _modal_viz_cache.get(viz_cache_key)
        if cached_json is not None:
            print(f"[CACHE HIT] Modal viz json - key: {viz_cache_key[:16]}...")
            return Response(content=cached_json, media_type="application/json")

        displacements = {nid: shapes[mode_index]
                         for nid, shapes in mode_shapes.items()}

        fig = generate_results_figure(
            structure_data,
            displacements,
            scale,
            theme=theme,
            animate=animate,
            frame_count=frame_count,
            interpolation_segments=detail,
        )
        fig_json = fig.to_json()
        _modal_viz_cache.set(viz_cache_key, fig_json)
        print(f"[CACHE SET] Modal viz json - key: {viz_cache_key[:16]}...")
        return Response(content=fig_json, media_type="application/json")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


_harmonic_viz_cache = LRUCache(50)


@app.post("/api/visualization/harmonic-results", summary="Visualizacion de respuesta armonica con el mismo motor grafico que estatico/modal")
async def get_harmonic_results_viz(
    request: HarmonicAnalysisRequest,
    frequency_hz: float = 1.0,
    scale: float = 1.0,
    theme: str = "dark",
):
    """Reusa generate_results_figure (el mismo codigo de estatico/modal) tomando
    la parte REAL del desplazamiento complejo a la frecuencia pedida (fase 0).
    Se pierde el aporte de la parte imaginaria (desfase entre grados de libertad
    que no oscilan en fase) a cambio de la paleta/curvatura identica a Estatico.
    """
    try:
        request_data = request.model_dump()
        analysis_cache_key = compute_structure_hash(
            request.structure.model_dump(),
            {
                "analysis": "harmonic",
                "freq_start": request.freq_start,
                "freq_end": request.freq_end,
                "num_points": request.num_points,
                "damping_ratio": request.damping_ratio,
                "is_unbalanced": request.is_unbalanced,
                "unbalanced_me": request.unbalanced_me,
                "unbalanced_node_id": request.unbalanced_node_id,
                "unbalanced_direction": request.unbalanced_direction,
                "unbalanced_plane": request.unbalanced_plane,
                "unbalanced_mass": request.unbalanced_mass,
                "unbalanced_eccentricity": request.unbalanced_eccentricity,
            },
        )

        cached = _harmonic_cache.get(analysis_cache_key)
        if cached is not None:
            print(f"[CACHE HIT] Harmonic analysis (viz) - key: {analysis_cache_key[:16]}...")
            result_dict = cached
        else:
            result_dict = run_harmonic_analysis(request_data)
            if "error" in result_dict:
                raise HTTPException(status_code=400, detail=result_dict["error"])
            _harmonic_cache.set(analysis_cache_key, result_dict)
            print(f"[CACHE SET] Harmonic analysis (viz) - key: {analysis_cache_key[:16]}...")

        frequencies = result_dict.get("frequencies_sweep") or []
        if not frequencies:
            raise HTTPException(status_code=400, detail="Sin barrido de frecuencias calculado")

        freq_idx = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - frequency_hz))

        viz_cache_key = compute_structure_hash(
            request.structure.model_dump(),
            {**{"analysis": "harmonic-viz", "freq_idx": freq_idx, "scale": round(float(scale), 8), "theme": theme}},
        )
        cached_json = _harmonic_viz_cache.get(viz_cache_key)
        if cached_json is not None:
            print(f"[CACHE HIT] Harmonic viz json - key: {viz_cache_key[:16]}...")
            return Response(content=cached_json, media_type="application/json")

        components = result_dict.get("node_displacement_components", {})
        displacements = {
            int(node_id): [
                comp["ux_real_m"][freq_idx], comp["uy_real_m"][freq_idx], comp["uz_real_m"][freq_idx],
                comp["rx_real_rad"][freq_idx], comp["ry_real_rad"][freq_idx], comp["rz_real_rad"][freq_idx],
            ]
            for node_id, comp in components.items()
        }

        structure_data = request.structure.model_dump()
        fig = generate_results_figure(structure_data, displacements, scale, theme=theme)
        fig_json = fig.to_json()
        _harmonic_viz_cache.set(viz_cache_key, fig_json)
        print(f"[CACHE SET] Harmonic viz json - key: {viz_cache_key[:16]}...")
        return Response(content=fig_json, media_type="application/json")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/projects")
async def list_projects(user=Depends(get_current_user)):
    try:
        response = supabase.table("projects").select("*").eq("user_id", user["sub"]).execute()
        return response.data
    except Exception as e:
        print(f"Error listing projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/{project_id}")
async def get_project(project_id: str, user=Depends(get_current_user)):
    try:
        response = supabase.table("projects").select("*").eq("id", project_id).eq("user_id", user["sub"]).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Project not found")
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects")
async def create_project(project_data: dict, user=Depends(get_current_user)):
    try:
        raw_structure = project_data.get("structure_data", {})
        validated_structure = StructureInput(**raw_structure).model_dump(by_alias=True)
        
        payload = {
            "name": project_data.get("name", "Nuevo Proyecto"),
            "structure_data": validated_structure,
            "user_id": user["sub"]
        }
        response = supabase.table("projects").insert(payload).execute()
        return response.data[0]
    except Exception as e:
        print(f"Error creating project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str, user=Depends(get_current_user)):
    try:
        response = supabase.table("projects").delete().eq("id", project_id).eq("user_id", user["sub"]).execute()
        return {"status": "deleted"}
    except Exception as e:
        print(f"Error deleting project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, project_data: dict, user=Depends(get_current_user)):
    try:
        update_payload = {}
        if "name" in project_data:
            update_payload["name"] = project_data["name"]
        
        if "structure_data" in project_data:
            raw_structure = project_data["structure_data"]
            validated_structure = StructureInput(**raw_structure).model_dump(by_alias=True)
            update_payload["structure_data"] = validated_structure

        if not update_payload:
            return {"status": "no changes"}

        response = supabase.table("projects").update(update_payload).eq("id", project_id).eq("user_id", user["sub"]).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Project not found or not owned by user")
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating project: {e}")
        raise HTTPException(status_code=500, detail=str(e))
