from typing import Optional, List
from dataclasses import dataclass

@dataclass
class Period:
    HOURLY = '1h'
    DAILY = "25h"
    WEEKLY = "7d"
    MONTHLY = "1m"
    YEARLY = "1y"