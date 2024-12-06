import logging

viz_logger = logging.getLogger("MctsVisualizationLogger")

VISUALIZATION = 25
logging.addLevelName(VISUALIZATION, "VISUALIZATION")

def log_visualization(self, message, *args, **kwargs):
    if self.isEnabledFor(VISUALIZATION):
        self._log(VISUALIZATION, message, args, **kwargs)

viz_logger.visualization = log_visualization
