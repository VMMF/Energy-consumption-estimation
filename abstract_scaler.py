from abc import ABC, abstractmethod


class AbstractScaler(ABC):

    

    @abstractmethod
    def learn():
        pass

    @abstractmethod
    def work(FWD = True):
        pass