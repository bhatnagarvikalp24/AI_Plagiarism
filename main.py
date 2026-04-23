import logging
import os

import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from config import get_settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NLTK data bootstrap (runs once at startup)
# ---------------------------------------------------------------------------
def _download_nltk_data():
    datasets = ["brown", "punkt", "averaged_perceptron_tagger"]
    for ds in datasets:
        try:
            nltk.data.find(f"corpora/{ds}" if ds != "punkt" else f"tokenizers/{ds}")
        except LookupError:
            logger.info("Downloading NLTK dataset: %s", ds)
            nltk.download(ds, quiet=True)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AI Book Detection API",
        description=(
            "Upload a PDF book and get a detailed analysis of how much of its "
            "content is likely AI-generated."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1", tags=["Detection"])

    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting AI Detection API...")
        os.makedirs(settings.upload_dir, exist_ok=True)
        _download_nltk_data()
        logger.info("Startup complete. LLM provider: %s", settings.llm_provider)

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
