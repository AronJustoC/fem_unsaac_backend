#!/usr/bin/env python3
"""
WSGI compatibility layer for Render deployment.
This file helps Render detect this as a Python web service.
"""

import os
import uvicorn
from main import app

# Import and run the FastAPI app with uvicorn
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)