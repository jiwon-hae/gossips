
from transformers import pipeline
from typing import List, Tuple
from enum import Enum
from transformers import pipeline
from nltk.tokenize import sent_tokenize

try:
    from ...models.events import Event, EVENT_TO_CATEGORY
except ImportError:
    import os
    import sys
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    sys.path.insert(0, project_root)

    from models.events import Event, EVENT_TO_CATEGORY


model_id = "dbmdz/bert-large-cased-finetuned-conll03-english"
hf_ner = pipeline("ner", model=model_id, tokenizer=model_id, aggregation_strategy="simple")
PRONOUNS = {"him", "her", "he", "she", "they", "them", "his", "hers", "their"}

zero_shot_event_classifier = pipeline(
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
    res = zero_shot_event_classifier(text, labels)
    pred = res['labels'][0]
    return EVENT_TO_CATEGORY[Event(pred)], Event(pred)


def extract_event(text: str):
    stakeholders = extract_people(text)
    event_type, event = _extract_event(text)

    return stakeholders, event_type, event
