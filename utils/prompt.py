from jinja2 import Environment, FileSystemLoader
from pathlib import Path


class PromptLoader:
    def __init__(self, path: str):
        prompt_path = Path(__file__).parent.parent / 'prompt' / path
        self.env = Environment(loader=FileSystemLoader(searchpath=prompt_path))

    def render(self, file_name: str, **params):
        template = self.env.get_template(file_name)
        return template.render(**params)
