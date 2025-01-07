from abc import ABC, abstractmethod


class Observer(ABC):
    @abstractmethod
    def notify(self, message: str):
        pass


class LoggingObserver(Observer):
    def __init__(self, log_file: str = None):
        self.log_file = log_file

    def notify(self, message: str):
        if self.log_file:
            with open(self.log_file, "a") as file:
                file.write(message + "\n")
        else:
            print(message)
