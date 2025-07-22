
from transformers import pipeline
from typing import List, Tuple
from enum import Enum
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from collections import Counter

try:
    from ..enums import Event, EVENT_TO_CATEGORY
    from ..chunker.chunk import DocumentChunk
    
except ImportError:
    import os
    import sys
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    sys.path.insert(0, project_root)
    
    from ingestion.enums import Event, EVENT_TO_CATEGORY
    from ingestion.chunker.chunk import DocumentChunk


ner_model_id = "dbmdz/bert-large-cased-finetuned-conll03-english"
hf_ner = pipeline("ner", model=ner_model_id, tokenizer=ner_model_id, aggregation_strategy="simple")
PRONOUNS = {"him", "her", "he", "she", "they", "them", "his", "hers", "their"}

sentiment_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
event_classifier = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli")



def extract_people(text: str) -> List[str]:
    ents = hf_ner(text)
    people = [
        ent["word"].strip()
        for ent in ents
        if ent["entity_group"] == "PER"
        and ent["word"].lower() not in PRONOUNS
    ]
    seen = set()
    return [p for p in people if not (p.lower() in seen or seen.add(p.lower()))]


def _extract_event(text: str) -> Tuple[Event, str]:
    labels = [e.value for e in Event]
    res = event_classifier(text, labels)
    pred = res['labels'][0]
    return EVENT_TO_CATEGORY[Event(pred)], Event(pred)


def extract_event(text: str):
    stakeholders = extract_people(text)
    event_type, event = _extract_event(text)

    return stakeholders, event_type, event


def aggregate_chunk_metadata(chunks: List[DocumentChunk], key : str) -> str:
    # Count the labels across all chunks
    counts = Counter(chunk.metadata[key] for chunk in chunks)
    # Pick the label with the highest count
    most_common, _ = counts.most_common(1)[0]
    return most_common


def extract_sentiment_from_chunks(chunks: List[DocumentChunk]):
    for chunk in chunks:
        sentiment = extract_sentiment(chunk.content)
        chunk.metadata['sentiment'] = sentiment
        
    doc_sentiment = aggregate_chunk_metadata(chunks, key = 'sentiment')
    return doc_sentiment

def extract_sentiment(chunk : str):
    result = sentiment_classifier(chunk)
    sentiment = result[0]['label']
    return sentiment


def extract_event_from_chunks(chunks : List[DocumentChunk]):
    for chunk in chunks:
        result = event_classifier(chunk.content, Event.values())
        event = result['labels'][0]
        chunk.metadata['event'] = event
    
    doc_event = aggregate_chunk_metadata(chunks, key = 'event')
    return doc_event