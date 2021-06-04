from abc import ABC, abstractmethod


class AbstractScaler(ABC):

    @abstractmethod
    def __init__(self,feature_range, scaler) -> None:
        self.feature_range = feature_range
        self.is_fit = False
        self.scaler = scaler
        super().__init__()

    @abstractmethod
    def learn():
        pass

    @abstractmethod
    def work(FWD = True):
        pass