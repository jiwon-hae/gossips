import wptools
import re
import logging
import mwparserfromhell

from typing import Optional
from dateutil import parser

try:
    from ..models.info import *
except ImportError:
    import os
    import sys
    
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

    from models.info import *

logger = logging.getLogger(__name__)

class WikipediaCollector:
    def __init__(self):
        pass

    def page(self, name: str):
        return wptools.page(name).get_parse()
    
    def info(self, name: str):
        page = self.page(name)
        return page.data.get("infobox", {})
    
    def _parse_spouse_field(self, raw: Optional[str]) -> List[Relationship]:
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
            
            name = name.replace("[", '').replace(']', '')
            name = name.split('|')[0]
            
            relations.append(
                Relationship(
                    partner=name,
                    start_yr=parser.parse(start_year).date(),
                    end_yr = parser.parse(end_year).date() if end_year else None,
                    relationship = RelationshipType(reason) if reason else RelationshipType(status)
                )
            )
        
        return relations
    
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

            
    def personal(self, name: str):
        info = self.info(name)
        return PersonalInfo(
            name = name ,
            spouse= self._parse_spouse_field(info.get('spouse', None)),
            occupation = self._parse_occupation(info.get("occupation", ''))
        )
    
if __name__ == '__main__':
    wikipediaCollector = WikipediaCollector()
    info = wikipediaCollector.personal("Justin Bieber")
    print(info)
    print('##########')
    
    info = wikipediaCollector.personal("Johnny Depp")
    print(info)
    print('##########')
    
    info = wikipediaCollector.personal("Jennifer Lopez")
    print(info)
    print('##########')
