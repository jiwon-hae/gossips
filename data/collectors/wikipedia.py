import wptools
import re
import logging
import mwparserfromhell
import wikipedia as wiki
from pathlib import Path

from typing import Optional
from dateutil import parser

try:
    from ...ingestion.models import *
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    ))
    
    from ingestion.models import *

logger = logging.getLogger(__name__)

class WikipediaCollector:
    def __init__(self):
        DATA_DIR = Path(__file__).resolve().parent.parent
        self.save_path = DATA_DIR / "wiki"
        self.save_path.mkdir(parents=True, exist_ok=True)

    def page(self, name: str):
        return wptools.page(name).get_parse()
    
    def info(self, name: str):
        page = self.page(name)
        
        info = page.data.get('infobox', {})
        info['title'] = page.data.get('title', '')
        return info
    
    def _parse_spouse_field(self, raw: Optional[str]) -> List[Relationship]:
        print(raw)
        if raw is None:
            return None
        
        template = re.findall(r"\{\{(.*?)\}\}", raw)
        relationship_history = template[1:] if template and template[0].startswith('ubl') else template
        relations = []
        
        for history in relationship_history:
            pattern = r'\[\[.*?\]\]|[^|]+'
            parts = re.findall(pattern, history)
            
            try:
                status, name, start_year, end_year, _, _, reason = parts
            except ValueError:
                status, name, start_year = parts
                end_year = None
                reason = None
            
            rel_str = reason or status
            rel_str_lower = rel_str.lower()
            
            if rel_str_lower in ['marriage', 'married']:
                relationship = RelationshipStatus.MARRIED
            elif rel_str_lower == 'divorced':
                relationship = RelationshipStatus.DIVORCED
            else:
                try:
                    relationship = RelationshipStatus(rel_str_lower)
                except ValueError:
                    relationship = RelationshipStatus.MARRIED  # Default to married
                
            name = name.replace("[", '').replace(']', '')
            name = name.split('|')[0]
            
            relations.append(
                Relationship(
                    partner=name,
                    start_yr=parser.parse(start_year).date(),
                    end_yr = parser.parse(end_year).date() if end_year else None,
                    relationship = relationship
                )
            )
        
        return relations

    def _parse_parents(self, raw):
        if not raw:
            return None
        
        matches = re.findall(r'\[\[([^|\]]+)(?:\|[^]]*)?\]\]', raw)
        return matches
        
        
    def _parse_family(self, raw):
        if not raw:
            return None
        
        print(raw)
    
    def _parse_occupation(self, raw):
        code = mwparserfromhell.parse(raw)
        occs = []

        # 1) Look for flatlist/hlist templates
        TAG_RE = re.compile(r"^<.*?>$")       # anything that’s just an HTML tag
   
        for tmpl in code.filter_templates():
            name = tmpl.name.strip_code().strip().lower()
            if name == "flatlist":
                # flatlist has a single parameter with "\n* item" lines
                flat = str(tmpl.params[0].value)
                for line in flat.splitlines():
                    line = line.strip()
                    
                    if line == '' or TAG_RE.match(line):
                        continue
                    
                    if line.startswith("*"):
                        occs.append(line[1:].strip().lower())
            elif name == "hlist":
                # hlist has multiple parameters
                for param in tmpl.params:
                    val = str(param.value).strip()
                    # skip citations or references
                    if val.lower().startswith("ref") or val.lower().startswith("cite"):
                        continue
                    
                    if val == '' or TAG_RE.match(val):
                        continue
                    
                    occs.append(val.lower())
    
        if not occs:
            occs = [raw.strip().lower()]

        cleaned = [mwparserfromhell.parse(o).strip_code().strip() for o in occs]
        return cleaned

            
    def profile(self, name: str) -> Optional[PersonalInfo]:
        try:
            info = self.info(name)
            return PersonalInfo(
                name = info.get('title', info.get('name', info.get('birth_name', name))),
                spouse= self._parse_spouse_field(info.get('spouse', None)),
                occupation = self._parse_occupation(info.get("occupation", '')),
                parents = self._parse_parents(info.get('parents', None))
            )
        except Exception as e:
            logger.error(f"Failed to retrieve profile of {name}: ({e})")
            return None
        
    def check_is_person(self, page):
        cats = [c.lower() for c in page.categories]
        return any(
            ("living people" in c) or
            ("births" in c and "births by year" not in c) or
            ("deaths" in c and "deaths by year" not in c)
            for c in cats
        )
        
    def save_wiki(self, name: str, retry: bool = False, auto_suggest: bool = False):
        try:
            
            wikipage = wiki.page(name, auto_suggest=auto_suggest)
            info = self.info(name)
            
            if not self.check_is_person(wikipage):
                return False
            
            path = Path(self.save_path, f"{info.get('title', info.get(name, info.get('birth_name', name)))}.md")
            path.write_text(wikipage.content, encoding='utf-8')
            logger.info(f"Successfully saved wiki of {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save wiki of {name}: ({e})")
            if retry:
                return False
            return self.save_wiki(name, retry=True, auto_suggest=True)
        
    
if __name__ == '__main__':
    wikipediaCollector = WikipediaCollector()
    info = wikipediaCollector.save_wiki("Nicolas Cage")
    print(info)
