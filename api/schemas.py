from pydantic import BaseModel, ConfigDict


class AnalysisReport(BaseModel):
    """Rapport final du Service 4, relayé tel quel (schéma souple)."""

    model_config = ConfigDict(extra="allow")
