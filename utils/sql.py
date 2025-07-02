from typing import Optional
from pathlib import Path

class SQLLoader:
    def __init__(self, base_dir: str = "sql"):
        self.base_path = Path(__file__).resolve().parents[1] / base_dir


    def load(self, *parts: str) -> str:
        path = self.base_path.joinpath(*parts)
        if not path.exists():
            raise FileNotFoundError(f"SQL file not found: {path}")
        return path.read_text().strip()


def get_sql_loader(base_dir: Optional[str] = None):
    return SQLLoader(base_dir or "sql")
