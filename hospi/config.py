from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Config:
	CHUNK_SIZE: int = 10000
	DATE_FORMAT: str = "%d/%m/%Y"

	SALES_THRESHOLDS: Dict[str, float] = None
	COLOR_SCHEME: Dict[str, str] = None

	def __post_init__(self):
		object.__setattr__(self, "SALES_THRESHOLDS", {
			"high_value": 100.0,
			"low_value": 10.0,
			"refund_threshold": -50.0,
		})
		object.__setattr__(self, "COLOR_SCHEME", {
			"primary": "#1f77b4",
			"secondary": "#ff7f0e",
			"negative": "#d62728",
		})
