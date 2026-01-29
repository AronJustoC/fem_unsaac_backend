"""
Compression and storage module for FEM analysis results.
Implements MessagePack serialization + Gzip compression for efficient storage.
"""
import gzip
import msgpack
from typing import Dict, Any, Optional
import io


def compress_result(result: Dict[str, Any]) -> bytes:
    """
    Compress analysis result using MessagePack + Gzip.
    
    Args:
        result: Dictionary containing analysis results (displacements, forces, frequencies, etc.)
    
    Returns:
        Compressed bytes ready for storage
    """
    # Serialize with MessagePack
    packed = msgpack.packb(result, use_bin_type=True)
    
    # Compress with Gzip (level 6 = balanced speed/compression)
    compressed = gzip.compress(packed, compresslevel=6)
    
    return compressed


def decompress_result(compressed_data: bytes) -> Dict[str, Any]:
    """
    Decompress and deserialize analysis result.
    
    Args:
        compressed_data: Gzip-compressed MessagePack data
    
    Returns:
        Deserialized result dictionary
    """
    # Decompress
    packed = gzip.decompress(compressed_data)
    
    # Deserialize
    result = msgpack.unpackb(packed, raw=False)
    
    return result


def upload_result(analysis_id: str, result: Dict[str, Any]) -> str:
    """
    Compress and upload analysis result to Supabase Storage.
    
    Args:
        analysis_id: UUID of the analysis
        result: Analysis result dictionary
    
    Returns:
        Public URL of the uploaded file
    
    Raises:
        Exception: If upload fails
    """
    from supabase_client import supabase
    
    compressed_data = compress_result(result)
    
    file_path = f"results/{analysis_id}.msgpack.gz"
    
    response = supabase.storage.from_("analysis-results").upload(
        path=file_path,
        file=compressed_data,
        file_options={"content-type": "application/octet-stream"}
    )
    
    public_url = supabase.storage.from_("analysis-results").get_public_url(file_path)
    
    return public_url


def download_result(analysis_id: str) -> Optional[Dict[str, Any]]:
    """
    Download and decompress analysis result from Supabase Storage.
    
    Args:
        analysis_id: UUID of the analysis
    
    Returns:
        Decompressed result dictionary, or None if not found
    """
    from supabase_client import supabase
    
    file_path = f"results/{analysis_id}.msgpack.gz"
    
    try:
        response = supabase.storage.from_("analysis-results").download(file_path)
        
        if not response:
            return None
        
        result = decompress_result(response)
        return result
    
    except Exception:
        return None


def delete_result(analysis_id: str) -> bool:
    """
    Delete analysis result from storage.
    
    Args:
        analysis_id: UUID of the analysis
    
    Returns:
        True if deletion succeeded, False otherwise
    """
    from supabase_client import supabase
    
    file_path = f"results/{analysis_id}.msgpack.gz"
    
    try:
        supabase.storage.from_("analysis-results").remove([file_path])
        return True
    except Exception:
        return False
