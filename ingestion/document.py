import logging
import fitz
import os
import glob

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)


def find_documents(document_folder: str, doc_patterns: Optional[List[str]]=None) -> List[str]:
    if not os.path.exists(document_folder):
        logger.error(f"Documents folder not found: {document_folder}")
        return []

    # TODO: apply patterns to get all the respective files
    patterns = doc_patterns if doc_patterns else ['*.txt']
    files = []

    for pattern in patterns:
        files.extend(glob.glob(os.path.join(document_folder, "**", pattern), recursive=True))
    return sorted(files)


def read_document(document_path: str) -> str:
    ext = Path(document_path).suffix.lower()

    if ext == '.pdf':
        return _read_pdf(document_path)
    elif ext == '.docx':
        return _read_docx(document_path)
    return _read_file(document_path)


def _read_pdf(document_path: str) -> str:
    """Read document content from pdf"""
    doc = fitz.open(document_path)
    content = "\n\n".join(page.get_text().strip() for page in doc)
    doc.close()
    return content


def _read_docx(document_path: str) -> str:
    """Read document content from DOCX and return as a single string."""
    doc = DocxDocument(document_path)
    content = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return content


def _read_file(document_path: str) -> List[str]:
    """Read document content from file"""
    try:
        with open(document_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(document_path, 'r', encoding='latin-1') as f:
            return f.read()


def extract_title(content: str, document_path: str) -> str:
    ext = Path(document_path).suffix.lower()

    if ext == '.pdf':
        return _extract_title_pdf(document_path)
    elif ext == '.docx':
        return _extract_title_docx(document_path)

    return _extract_title(content, document_path)


def _extract_title(content: str, file_path: str) -> str:
    """Extract document title from the document"""
    lines = content.split('\n')
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if line.startswith('# '):
            return line[2:].strip()

    # Fallback to filename
    return os.path.splitext(os.path.basename(file_path))[0]


def _extract_title_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    metadata = doc.metadata
    title = metadata.get("title", os.path.splitext(
        os.path.basename(file_path))[0]).strip()
    doc.close()
    return title


def _extract_title_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)
    title = doc.core_properties.title
    return title.strip() if title else os.path.splitext(os.path.basename(file_path))[0].strip()


def extract_document_metadata(content: str, document_path: str) -> Dict[str, Any]:
    """Extract metadata from the document"""
    ext = Path(document_path).suffix.lower()

    if ext == '.pdf':
        return _extract_metadata_pdf(content, document_path)
    elif ext == '.docx':
        return _extract_metadata_docx(content, document_path)
    elif ext == 'md':
        return _extract_metadata_md(content, document_path)

    metadata = {
        "file_path": document_path,
        "file_size": len(content),
        "ingestion_date": datetime.now().isoformat()
    }

    lines = content.split('\n')
    metadata['line_count'] = len(lines)
    metadata['word_count'] = len(content.split())

    return metadata


def _extract_metadata_pdf(content: str, file_path: str) -> Dict[str, Any]:
    doc = fitz.open(file_path)
    metadata = {
        "file_path": file_path,
        "file_size": os.path.getsize(file_path),
        "ingestion_date": datetime.now().isoformat(),
        "content_length": len(content),
        "word_count": len(content.split()),
        "line_count": len(content.split('\n')),
        "page_count": doc.page_count
    }

    pdf_metadata = doc.metadata
    if pdf_metadata:
        if pdf_metadata.get("title"):
            metadata["title"] = pdf_metadata["title"]
        if pdf_metadata.get("author"):
            metadata["author"] = pdf_metadata["author"]
        if pdf_metadata.get("subject"):
            metadata["subject"] = pdf_metadata["subject"]
        if pdf_metadata.get("creator"):
            metadata["creator"] = pdf_metadata["creator"]
        if pdf_metadata.get("producer"):
            metadata["producer"] = pdf_metadata["producer"]
        if pdf_metadata.get("creationDate"):
            metadata["creation_date"] = pdf_metadata["creationDate"]
        if pdf_metadata.get("modDate"):
            metadata["modification_date"] = pdf_metadata["modDate"]

    doc.close()
    return metadata


def _extract_metadata_docx(content: str, file_path: str) -> Dict[str, Any]:
    doc = DocxDocument(file_path)
    metadata = {
        "file_path": file_path,
        "file_size": os.path.getsize(file_path),
        "ingestion_date": datetime.now().isoformat(),
        "content_length": len(content),
        "word_count": len(content.split()),
        "line_count": len(content.split('\n')),
        "paragraph_count": len(doc.paragraphs)
    }

    core_props = doc.core_properties
    if core_props.title:
        metadata["title"] = core_props.title
    if core_props.author:
        metadata["author"] = core_props.author
    if core_props.subject:
        metadata["subject"] = core_props.subject
    if core_props.created:
        metadata["creation_date"] = core_props.created.isoformat()
    if core_props.modified:
        metadata["modification_date"] = core_props.modified.isoformat()
    if core_props.last_modified_by:
        metadata["last_modified_by"] = core_props.last_modified_by
    if core_props.category:
        metadata["category"] = core_props.category
    if core_props.comments:
        metadata["comments"] = core_props.comments
    if core_props.keywords:
        metadata["keywords"] = core_props.keywords
    if core_props.language:
        metadata["language"] = core_props.language

    return metadata


def _extract_metadata_md(content: str, file_path) -> Dict[str, Any]:
    """Extract metadata from document content."""
    metadata = {
        "file_path": file_path,
        "file_size": len(content),
        "ingestion_date": datetime.now().isoformat()
    }

    # Try to extract YAML frontmatter
    if content.startswith('---'):
        try:
            import yaml
            end_marker = content.find('\n---\n', 4)
            if end_marker != -1:
                frontmatter = content[4:end_marker]
                yaml_metadata = yaml.safe_load(frontmatter)
                if isinstance(yaml_metadata, dict):
                    metadata.update(yaml_metadata)
        except ImportError:
            logger.warning(
                "PyYAML not installed, skipping frontmatter extraction")
        except Exception as e:
            logger.warning(f"Failed to parse frontmatter: {e}")

    # Extract some basic metadata from content
    lines = content.split('\n')
    metadata['line_count'] = len(lines)
    metadata['word_count'] = len(content.split())

    return metadata
