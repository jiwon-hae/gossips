# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) system built with LlamaIndex, designed to work with both local and cloud-based language models. The project implements document ingestion, chunking, vector storage with PostgreSQL, and provides both agentic RAG capabilities and API endpoints.

## Architecture

### Core Components

- **ingestion/**: Document processing pipeline with chunking and graph building capabilities
  - `ingest.py`: Main ingestion pipeline with `DocumentIngestionPipeline` class
  - `chunker.py`: Document chunking with configurable parameters (`ChukingConfig`, `DocumentChunk`)
  - `graph_builder.py`: Graph construction using Graphiti

- **backend/**: FastAPI-based API layer
  - `apis/request.py`: API router setup
  - `services/`: Service layer (currently empty structure)

- **agent/**: Agentic capabilities
  - `providers.py`: Agent provider implementations (currently empty)

- **utils/**: Shared utilities
  - `secrets.py`: Environment variable management with `SecretsLoader` class

- **sql/**: Database schema and functions
  - `schema.sql`: PostgreSQL schema with pgvector extension for embeddings

### Database Schema

The PostgreSQL database uses the pgvector extension and includes:
- `documents`: Document metadata and content storage
- `chunks`: Document chunks with vector embeddings (1536 dimensions)
- `sessions`: User session management
- `messages`: Chat message history

Key functions:
- `match_chunks()`: Vector similarity search
- `hybrid_search()`: Combined vector and text search
- `get_document_chunks()`: Retrieve chunks by document ID

### Configuration

- Environment variables managed through `utils/secrets.py`
- `.env` file exists for local configuration
- No package manager files detected (requirements.txt, pyproject.toml, package.json)

## Development Workflow

Since no package manager configuration files were found, you'll need to:

1. **Environment Setup**: Use the `SecretsLoader` class to load environment variables from `.env`
2. **Database Setup**: Run `sql/schema.sql` to initialize PostgreSQL with pgvector
3. **Local Models**: The project supports both local models (Hugging Face transformers) and cloud APIs

## Key Classes and Interfaces

- `DocumentIngestionPipeline`: Main ingestion orchestrator
- `ChukingConfig`: Chunking configuration with validation
- `DocumentChunk`: Chunk data structure
- `SecretsLoader`: Environment variable management
- `GraphBuilder`: Graph construction (uses Graphiti)

## Notes

- Typos exist in class names (`ChukingConfig` should be `ChunkingConfig`, `chunk_overalp` should be `chunk_overlap`)
- The notebooks demonstrate integration with both local Hugging Face models and OpenAI
- FastAPI backend structure is initialized but minimal
- Agent provider system is scaffolded but not implemented