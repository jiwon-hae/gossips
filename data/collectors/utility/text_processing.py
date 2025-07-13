
from transformers import pipeline
from typing import List, Tuple
from enum import Enum
from transformers import pipeline
from nltk.tokenize import sent_tokenize

try:
    from ..events import *
except ImportError:
    import os, sys
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    
    from collectors.events import *
    

model_id = "dbmdz/bert-large-cased-finetuned-conll03-english"
hf_ner = pipeline("ner", model=model_id, grouped_entities=True)
PRONOUNS = {"him", "her", "he", "she", "they", "them", "his", "hers", "their"}

zero_shot_event_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    


def extract_people(text: str) -> List[str]:
    ents = hf_ner(text)
    people = {
        e["word"]
        for e in ents
        if e["entity_group"] == "PER"
        and e["word"].lower() not in PRONOUNS
    }
    return sorted(people)

def _extract_event(text: str) -> Tuple[EventType, str]:
    labels = [e.value for e in EventType]
    print(labels)
    res = zero_shot_event_classifier("Justin and Hailey are getting a divorce.", labels)
    print(res)
    print(res['labels'][0])
    return None, None
    


def extract_event(text: str):
    stakeholders = extract_people(text)
    event_type, event = _extract_event(text)
    

_extract_event("Justin and Hailey are getting a divorce")
