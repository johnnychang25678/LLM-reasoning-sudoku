from enum import Enum
from pprint import pformat
class VisualizationEventType(Enum):
    STARTING_SEARCH = "Starting search"
    STARTING_ITERATION = "Starting iteration"
    SELECTION = "Selection"
    EXPANSION = "Expansion"
    ROLLOUT = "Rollout"
    BACKPROPAGATION = "Backpropagation"
    COMPLETED_SEARCH = "Completed search"

class VisualizationEvent:
    def __init__(self, event_type: VisualizationEventType, data: dict):
        self.event_type = event_type
        self.data = data

    def __str__(self):
        return f"{self.event_type.value}: {pformat(self.data)}"
    
    
