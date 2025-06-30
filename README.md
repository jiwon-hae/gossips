# 💫 Gossips - Celebrity News Intelligence Platform

**Gossips** is an AI-powered platform that automatically collects, analyzes, and serves the latest celebrity news, gossip, conflicts, and events. Built with LlamaIndex and advanced machine learning, it provides intelligent insights into celebrity relationships, scandals, feuds, and trending topics.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### 🔍 **Intelligent News Collection**
- **Multi-source scraping** from Google News and celebrity-focused outlets
- **Real-time monitoring** of breaking celebrity news
- **Automated content filtering** for relevance and quality
- **Duplicate detection** and content deduplication

### 🤖 **AI-Powered Classification**
- **Advanced ML models** (Scikit-learn + PyTorch) for event classification
- **25+ event categories** including relationships, feuds, scandals, controversies
- **Multi-framework support**: Random Forest, LSTM, Transformers, MLP
- **Comprehensive model evaluation** with statistical significance testing

### 📊 **Event Categories**
- **Relationships**: Divorce, Breakup, Engagement, Marriage, Dating, Cheating
- **Conflicts**: Feuds, Fights, Lawsuits, Controversies, Scandals, Beef, Diss
- **Personal**: Pregnancy, Health Issues, Addiction, Mental Health
- **Career**: New Projects, Awards, Collaborations, Milestones
- **Legal**: Court Cases, Settlements, Arrests
- **Business**: Ventures, Endorsements, Financial Issues

### 🚀 **RAG-Powered Query Engine**
- **Retrieval-Augmented Generation** for intelligent celebrity gossip queries
- **Vector embeddings** with LRU caching for fast similarity search
- **Graph-based relationships** between celebrities and events
- **PostgreSQL + pgvector** for scalable vector storage

### 📈 **Analytics & Insights**
- **Trend analysis** of celebrity events over time
- **Sentiment tracking** for public perception analysis
- **Relationship mapping** between celebrities
- **Event correlation** and pattern detection

## 🏗️ Architecture

```
gossips/
├── 📊 data/
│   ├── collection/          # News scraping & data collection
│   ├── predictor/          # ML models & training pipeline
│   └── training/           # Training data & results
├── 🧠 ingestion/
│   ├── chunker/            # Document chunking strategies
│   ├── embed/              # Embeddings with LRU cache
│   └── graph/              # Knowledge graph construction
├── 🤖 agent/
│   ├── providers.py        # LLM & embedding providers
│   └── graph/              # Graph-based agents
├── 🌐 backend/
│   └── apis/               # FastAPI endpoints
├── 💾 sql/
│   └── schema.sql          # PostgreSQL + pgvector schema
└── 📓 notebooks/           # Analysis & experimentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL with pgvector extension
- OpenAI API key (or local models)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gossips.git
cd gossips
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment**
```bash
cp .env.example .env
# Edit .env with your API keys and database credentials
```

4. **Initialize database**
```bash
psql -d your_database -f sql/schema.sql
```

### 🎯 Usage Examples

#### **Collect Celebrity News**
```bash
# Collect latest celebrity news
python data/collection/collect.py --max-articles 1000

# Run full data pipeline with classification
python data/collection/pipeline.py --categories "feud,divorce,scandal"
```

#### **Train ML Models**
```bash
# Train traditional ML models
python data/predictor/train.py --models random_forest naive_bayes

# Train PyTorch neural networks
python data/predictor/train.py --models mlp lstm transformer

# Comprehensive evaluation with statistical testing
python data/predictor/train.py --models random_forest mlp --comprehensive-eval
```

#### **Query Celebrity Gossip**
```python
from ingestion.ingest import DocumentIngestionPipeline
from agent.providers import get_chat_client

# Initialize RAG pipeline
pipeline = DocumentIngestionPipeline()
documents = pipeline.process_directory("data/collected_news/")

# Query the gossip database
client = get_chat_client()
response = client.chat.completions.create(
    messages=[{
        "role": "user", 
        "content": "What are the latest feuds between Taylor Swift and other celebrities?"
    }]
)
```

## 🔧 Configuration

### **Data Collection**
```python
# data/collection/config.py
@dataclass
class DataCollectionConfig:
    max_articles_per_run: int = 1000
    min_confidence_threshold: float = 0.7
    enable_synthetic_augmentation: bool = True
    target_celebrities: List[str] = field(default_factory=list)
```

### **ML Training**
```python
# data/predictor/pytorch_classifier.py
@dataclass
class PyTorchTrainingConfig:
    model_type: str = "transformer"  # mlp, lstm, transformer
    embedding_dim: int = 128
    hidden_dim: int = 256
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
```

### **Embedding Cache**
```python
# ingestion/embed/embedder.py
embedder = create_embedder(
    model="text-embedding-3-small",
    use_cache=True,
    cache_size=1000
)
```

## 📊 Database Schema

```sql
-- Core tables with pgvector support
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    url TEXT UNIQUE,
    published_date TIMESTAMP,
    source VARCHAR(100),
    celebrities TEXT[],
    event_category VARCHAR(50)
);

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    content TEXT NOT NULL,
    embedding vector(1536),  -- pgvector for similarity search
    chunk_index INTEGER,
    metadata JSONB
);

-- Vector similarity search
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);
```

## 🧪 Testing

### **Run Tests**
```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/ingestion/chunker/

# Run with coverage
pytest --cov=ingestion --cov-report=html
```

### **Model Evaluation**
```bash
# Generate comprehensive evaluation report
python data/predictor/train.py --comprehensive-eval --models all

# View results
open data/predictor/evaluation_results/model_comparison.png
```

## 📈 Performance Optimizations

### **LRU Caching**
- **Embedding cache** using heapq for O(log n) eviction
- **95%+ cache hit rate** for similar queries
- **Memory efficient** with automatic cleanup

### **Batch Processing**
- **Parallel embedding generation** for large document sets
- **Optimized chunking** with semantic boundaries
- **Rate limiting** and retry logic for API calls

### **Vector Search**
- **PostgreSQL pgvector** for scalable similarity search
- **Hybrid search** combining vector and text matching
- **Index optimization** for sub-second query times

## 🛣️ Roadmap

- [ ] **Real-time streaming** for breaking celebrity news
- [ ] **Social media integration** (Twitter, Instagram APIs)
- [ ] **Celebrity sentiment tracking** over time
- [ ] **Web dashboard** with interactive analytics
- [ ] **API rate limiting** and user authentication
- [ ] **Multi-language support** for international celebrities
- [ ] **Mobile app** for gossip notifications

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run linting
black . && flake8 . && mypy .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LlamaIndex** for the RAG framework
- **PyTorch** for neural network implementations
- **pgvector** for efficient vector storage
- **OpenAI** for embedding and language models
- **Google News** for celebrity news data

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/gossips/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/gossips/discussions)

---

**Built with ❤️ for celebrity gossip enthusiasts and AI developers**

*Gossips - Where AI meets celebrity culture* 🌟
