"""
Autonomous Constellation Manager (ACM)
National Space Hackathon 2026
Main FastAPI application - exposes all required REST endpoints on port 8000
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api.telemetry import router as telemetry_router
from api.maneuver import router as maneuver_router
from api.simulate import router as simulate_router
from api.visualization import router as visualization_router
from state.fleet import fleet_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("acm")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the ACM system on startup."""
    logger.info("ACM system starting up...")
    fleet_state.initialize()
    logger.info(f"Fleet initialized with {len(fleet_state.satellites)} satellites")
    yield
    logger.info("ACM system shutting down.")


app = FastAPI(
    title="Autonomous Constellation Manager",
    description="NSH 2026 - Orbital Debris Avoidance & Constellation Management",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all API routers
app.include_router(telemetry_router, prefix="/api")
app.include_router(maneuver_router, prefix="/api")
app.include_router(simulate_router, prefix="/api")
app.include_router(visualization_router, prefix="/api")

# Serve the frontend dashboard
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

