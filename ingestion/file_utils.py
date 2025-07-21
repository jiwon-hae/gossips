import logging
import json
import os
import glob
import pandas as pd


from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Literal
from datetime import datetime
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)

def find_documents(document_folder: str, doc_patterns: Optional[List[str]] = None) -> List[str]:
    if not os.path.exists(document_folder):
        logger.error(f"Documents folder not found: {document_folder}")
        return []

    # Handle both extension patterns (.txt) and glob patterns (*.txt)
    patterns = doc_patterns or ['*.txt']
    normalized_patterns = []
    
    for pattern in patterns:
        if pattern.startswith('.'):
            # Convert extension to glob pattern: .txt -> *.txt
            normalized_patterns.append(f"*{pattern}")
        else:
            # Already a glob pattern
            normalized_patterns.append(pattern)
    
    files = []
    for pattern in normalized_patterns:
        files.extend(glob.glob(os.path.join(
            document_folder, "**", pattern), recursive=True))
    
    return sorted(files)

