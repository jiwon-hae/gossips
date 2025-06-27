class IngestinConfig:
    """
    Configuration class for the document ingestion pipeline.
    
    This class serves as a placeholder for ingestion configuration parameters.
    Currently empty, but can be extended with specific configuration options
    such as supported file types, processing settings, etc.
    """
    pass


class DocumentIngestionPipeline:
    """
    Main pipeline for ingesting documents into the RAG system.
    
    This class orchestrates the document ingestion process, including
    loading documents, processing them through the chunking pipeline,
    and storing the results for retrieval.
    
    Attributes:
        config (IngestinConfig): Configuration for the ingestion process
        document_folder (str): Path to the folder containing documents to ingest
        clean_before_ingest (bool): Whether to clean existing data before ingestion
    """
    
    def __init__(self, config: IngestinConfig, document_folder: str = 'documents', clean_before_ingest: bool = False):
        """
        Initialize the document ingestion pipeline.
        
        Args:
            config (IngestinConfig): Configuration for the ingestion process
            document_folder (str): Path to the folder containing documents. Default: 'documents'
            clean_before_ingest (bool): Whether to clean existing data before ingestion. Default: False
        """
        self.config = config
        self.document_folder = document_folder
        self.clean_before_ingest = clean_before_ingest
        
        
        