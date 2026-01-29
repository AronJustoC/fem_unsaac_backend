import pytest
import numpy as np
from storage import compress_result, decompress_result


def test_compress_decompress_simple():
    result = {
        "displacements": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "forces": [[10.0, 20.0], [30.0, 40.0]]
    }
    
    compressed = compress_result(result)
    decompressed = decompress_result(compressed)
    
    assert decompressed == result
    assert isinstance(compressed, bytes)
    assert len(compressed) < len(str(result).encode())


def test_compress_decompress_with_numpy():
    result = {
        "displacements": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).tolist(),
        "frequencies": [1.5, 2.3, 4.7, 8.1],
        "mode_shapes": np.random.rand(100, 10).tolist()
    }
    
    compressed = compress_result(result)
    decompressed = decompress_result(compressed)
    
    assert decompressed.keys() == result.keys()
    assert np.allclose(decompressed["displacements"], result["displacements"])
    assert np.allclose(decompressed["frequencies"], result["frequencies"])
    assert np.allclose(decompressed["mode_shapes"], result["mode_shapes"])


def test_compression_ratio():
    large_result = {
        "displacements": np.random.rand(1000, 6).tolist(),
        "forces": np.random.rand(1000, 6).tolist(),
        "frequencies": list(range(100)),
        "metadata": {"nodes": 1000, "elements": 2000}
    }
    
    import json
    json_size = len(json.dumps(large_result).encode())
    compressed = compress_result(large_result)
    compressed_size = len(compressed)
    
    compression_ratio = json_size / compressed_size
    assert compression_ratio > 2, f"Expected >2x compression, got {compression_ratio:.2f}x"


def test_empty_result():
    result = {}
    compressed = compress_result(result)
    decompressed = decompress_result(compressed)
    assert decompressed == result


def test_nested_structure():
    result = {
        "analysis": {
            "type": "modal",
            "modes": 12,
            "results": {
                "frequencies": [1.0, 2.0, 3.0],
                "damping": [0.05, 0.05, 0.05]
            }
        },
        "metadata": {
            "timestamp": "2026-01-27",
            "version": "1.0"
        }
    }
    
    compressed = compress_result(result)
    decompressed = decompress_result(compressed)
    assert decompressed == result
