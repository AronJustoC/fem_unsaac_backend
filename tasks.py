import time
from supabase_client import supabase
from analysis_core.api_adapters import run_static_analysis, run_modal_analysis
from storage import upload_result

async def run_analysis_task(analysis_id: str, structure_data: dict, analysis_type: str, num_modes: int = 12):
    try:
        supabase.table("analyses").update({"status": "running", "progress": 10}).eq("id", analysis_id).execute()
        
        start_time = time.time()
        if analysis_type == "static":
            result = run_static_analysis(structure_data)
        elif analysis_type == "modal":
            result = run_modal_analysis(structure_data, num_modes)
        else:
            raise ValueError("Unsupported analysis type")
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        if "error" in result:
            supabase.table("analyses").update({
                "status": "failed",
                "error_message": result["error"],
                "progress": 100
            }).eq("id", analysis_id).execute()
            return

        result_url = upload_result(analysis_id, result)
        
        supabase.table("analyses").update({
            "status": "done",
            "progress": 100,
            "computation_time_ms": duration_ms,
            "result_url": result_url
        }).eq("id", analysis_id).execute()

    except Exception as e:
        supabase.table("analyses").update({
            "status": "failed",
            "error_message": str(e),
            "progress": 100
        }).eq("id", analysis_id).execute()
