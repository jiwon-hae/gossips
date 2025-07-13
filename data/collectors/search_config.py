from typing import Optional, List
from dataclasses import dataclass

# try:
#     from .category import EventCategory
# except ImportError:
#     import sys
#     import os
#     sys.path.append(os.path.dirname(
#         os.path.dirname(os.path.abspath(__file__))))
    
#     from category import EventCategory

@dataclass
class Period:
    HOURLY = '1h'
    DAILY = "25h"
    WEEKLY = "7d"
    MONTHLY = "1m"
    YEARLY = "1y"