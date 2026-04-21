import json
from core.utils.json_stream import StreamJSONParser

def test_stream_parser():
    parser = StreamJSONParser(target_key="tasks")
    
    # Simulating chunks from an LLM
    chunks = [
        '{"reasoning": "I need to do things.", "tasks": [',
        '{"id": "task_1", "description": "task 1", ',
        '"required_capabilities": ["search"], "expected_output": "out1"}, ',
        '{"id": "task_2", "description": "task 2", ',
        '"required_capabilities": ["developer"], "expected_output": "out2"}',
        ']}'
    ]
    
    found_tasks = []
    for chunk in chunks:
        for obj in parser.feed(chunk):
            print(f"Parsed Object: {obj['id']}")
            found_tasks.append(obj)
            
    assert len(found_tasks) == 2
    assert found_tasks[0]['id'] == 'task_1'
    assert found_tasks[1]['id'] == 'task_2'
    print("Stream Parser Test Passed!")

if __name__ == "__main__":
    test_stream_parser()
