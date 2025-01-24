from pydantic import BaseModel

class BenchmarkMetrics(BaseModel):
    latency: float
    energy_usage: float
    gwp: float
    mrr: float
    recall: float