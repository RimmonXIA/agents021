import json
import re
from typing import Generator, Any

class StreamJSONParser:
    """
    A lightweight parser to extract individual objects from a streaming JSON list.
    Designed for patterns like: {"tasks": [{"id": "t1", ...}, {"id": "t2", ...}]}
    """
    def __init__(self, target_key: str = "tasks"):
        self.target_key = target_key
        self.buffer = ""
        self.in_list = False
        self.depth = 0
        self.obj_buffer = ""

    def feed(self, chunk: str) -> Generator[dict[str, Any], None, None]:
        """Feeds a chunk of text and yields completed objects."""
        self.buffer += chunk
        
        # Check if we've reached the start of the target list
        if not self.in_list:
            match = re.search(fr'"{self.target_key}"\s*:\s*\[', self.buffer)
            if match:
                self.in_list = True
                self.buffer = self.buffer[match.end():]
        
        if self.in_list:
            for char in self.buffer:
                self.obj_buffer += char
                if char == '{':
                    self.depth += 1
                elif char == '}':
                    self.depth -= 1
                    
                    if self.depth == 0:
                        # Attempt to parse the completed object
                        try:
                            # Strip leading commas or whitespace
                            clean_obj = self.obj_buffer.strip().lstrip(',')
                            obj = json.loads(clean_obj)
                            yield obj
                            self.obj_buffer = ""
                        except json.JSONDecodeError:
                            # Not quite a full object or malformed, continue buffering
                            pass
                elif char == ']' and self.depth == 0:
                    # End of the target list
                    self.in_list = False
                    break
            
            # Keep only the part of the buffer that hasn't been processed into obj_buffer
            self.buffer = ""
