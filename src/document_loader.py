"""Document loading.

Each file format has a small loader function registered with ``@register_loader``.
To support a new format (e.g. Markdown) you only write one function and register it:

    @register_loader(".md")
    def _load_markdown(path: Path) -> LoadedContent: ...

Loaders return LangChain ``Document`` objects, one per page (PDF), per block of rows
(CSV) or one for the whole file (TXT, DOCX). ``load_document`` then stamps every
section with the metadata used later for citations: source, file_type, doc_id, page.

We use pypdf / python-docx / csv directly instead of ``langchain-community`` loaders,
because that package is being sunset and these libraries give us page-level control.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from docx import Document as DocxDocument
from langchain_core.documents import Document
from pypdf import PdfReader

from src.exceptions import CorruptedDocumentError, EmptyDocumentError, RAGError, UnsupportedFileTypeError

logger = logging.getLogger(__name__)

CSV_ROWS_PER_SECTION = 50


@dataclass
class LoadedContent:
    """Raw output of a format-specific loader."""

    sections: list[Document]
    page_count: int = 0  # 0 = the format has no page concept


@dataclass
class LoadedDocument:
    """A fully loaded document ready for cleaning and chunking."""

    filename: str
    file_type: str
    doc_id: str
    sections: list[Document] = field(default_factory=list)
    page_count: int = 0


LoaderFunc = Callable[[Path], LoadedContent]
_LOADERS: dict[str, LoaderFunc] = {}


def register_loader(*extensions: str) -> Callable[[LoaderFunc], LoaderFunc]:
    """Register a loader function for one or more file extensions."""

    def decorator(func: LoaderFunc) -> LoaderFunc:
        for extension in extensions:
            _LOADERS[extension.lower()] = func
        return func

    return decorator


def supported_extensions() -> tuple[str, ...]:
    """Extensions that currently have a registered loader."""
    return tuple(sorted(_LOADERS))


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


@register_loader(".pdf")
def _load_pdf(path: Path) -> LoadedContent:
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise CorruptedDocumentError("The PDF is password-protected and cannot be read.")
        sections = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                sections.append(Document(page_content=text, metadata={"page": page_number}))
        return LoadedContent(sections=sections, page_count=len(reader.pages))
    except RAGError:
        raise
    except Exception as exc:  # pypdf raises many different error types
        raise CorruptedDocumentError() from exc


@register_loader(".docx")
def _load_docx(path: Path) -> LoadedContent:
    try:
        document = DocxDocument(str(path))
        lines = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    lines.append(" | ".join(cells))
        text = "\n".join(lines)
        sections = [Document(page_content=text, metadata={})] if text.strip() else []
        return LoadedContent(sections=sections, page_count=0)
    except RAGError:
        raise
    except Exception as exc:
        raise CorruptedDocumentError() from exc


@register_loader(".txt")
def _load_txt(path: Path) -> LoadedContent:
    try:
        text = _decode(path.read_bytes())
    except OSError as exc:
        raise CorruptedDocumentError() from exc
    sections = [Document(page_content=text, metadata={})] if text.strip() else []
    return LoadedContent(sections=sections, page_count=0)


@register_loader(".csv")
def _load_csv(path: Path) -> LoadedContent:
    try:
        text = _decode(path.read_bytes())
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(io.StringIO(text), dialect=dialect)

        sections: list[Document] = []
        block: list[str] = []
        block_start = 1
        row_number = 0
        for row_number, row in enumerate(reader, start=1):
            fields = [f"{key}: {value}" for key, value in row.items() if key and value]
            if fields:
                block.append(f"Row {row_number}: " + "; ".join(fields))
            if row_number % CSV_ROWS_PER_SECTION == 0:
                if block:
                    sections.append(_csv_section(block, block_start, row_number))
                block, block_start = [], row_number + 1
        if block:
            sections.append(_csv_section(block, block_start, row_number))
        return LoadedContent(sections=sections, page_count=0)
    except RAGError:
        raise
    except Exception as exc:
        raise CorruptedDocumentError() from exc


def _csv_section(lines: list[str], start: int, end: int) -> Document:
    return Document(page_content="\n".join(lines), metadata={"rows": f"{start}-{end}"})


def load_document(path: Path, *, filename: str, doc_id: str) -> LoadedDocument:
    """Load a file from disk and attach citation metadata to every section."""
    extension = path.suffix.lower()
    loader = _LOADERS.get(extension)
    if loader is None:
        raise UnsupportedFileTypeError(
            f"'{filename}' is not supported. Allowed types: {', '.join(supported_extensions())}."
        )

    content = loader(path)
    if not content.sections:
        raise EmptyDocumentError(f"'{filename}' does not contain any readable text.")

    file_type = extension.lstrip(".")
    for section in content.sections:
        section.metadata.update({"source": filename, "file_type": file_type, "doc_id": doc_id})

    logger.info(
        "Document processed: %s (%s) -> %d section(s), %d page(s)",
        filename,
        file_type,
        len(content.sections),
        content.page_count,
    )
    return LoadedDocument(
        filename=filename,
        file_type=file_type,
        doc_id=doc_id,
        sections=content.sections,
        page_count=content.page_count,
    )
