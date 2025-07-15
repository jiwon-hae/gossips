
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
hf_ner = pipeline("ner", model=model_id, grouped_entities=True)
PRONOUNS = {"him", "her", "he", "she", "they", "them", "his", "hers", "their"}

zero_shot_event_classifier = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli")


def extract_people(text: str) -> List[str]:
    ents = hf_ner(text)
    people = {
        e["word"]
        for e in ents
        if e["entity_group"] == "PER"
        and e["word"].lower() not in PRONOUNS
    }
    return sorted(people)


def _extract_event(text: str) -> Tuple[Event, str]:
    labels = [e.value for e in Event]
    res = zero_shot_event_classifier(text, labels)
    pred = res['labels'][0]
    return EVENT_TO_CATEGORY[Event(pred)], Event(pred)


def extract_event(text: str):
    stakeholders = extract_people(text)
    event_type, event = _extract_event(text)

    return stakeholders, event_type, event
