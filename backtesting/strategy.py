import abc
import numpy as np
from typing import Callable

class Strategy(metaclass = abc.ABCMeta):
    '''
    strategy.init
    strategy.next
    '''

    def __init__(self, broker,data):
        '''
        build the strategy object

        '''
        self._indicators = []
        self._broker = broker
        self._data = data
        self.tick = 0

    def I(self, func: callable, *args) -> np.ndarray:
        '''
        import the predicted index here
        decide buy or sell 
        '''
        value = func(*args)
        value = np.asarray(value)
        assert_msg(value.shape[-1] == len(self._data.Close))

        self._indicators.append(value)
        return value
    
    @property
    def tick(self):
        return self._tick
    
    @abc.abstractmethod
    def init(self):
        pass

    @abc.abstractmethod
    def next(self, next):
        pass
    



