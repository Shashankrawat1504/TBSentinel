"""Main entry point for running the FastAPI inference server.

The server binds to loopback (``127.0.0.1``) by default because the inference
API has no authentication — anything that can reach the port can call
``/predict`` and ``/explain``. Host, port, reload and log level come from the
``api:`` block of ``configs/inference/inference.yaml``; the environment
variables ``TBSENTINEL_API_HOST`` and ``TBSENTINEL_API_PORT`` override them for
a deliberate LAN demo, e.g.::

    TBSENTINEL_API_HOST=0.0.0.0 python -m src.inference.main
"""

import uvicorn

from src.inference.config import api_settings

if __name__ == "__main__":
    settings = api_settings()
    uvicorn.run(
        "src.inference.api:app",
        host=settings["host"],
        port=settings["port"],
        reload=settings["reload"],
        log_level=settings["log_level"],
    )
