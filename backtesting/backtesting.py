'''
This backtest uses a vectorized backtesting framework, and because this framework 
typically utilizes OHLC data, the simulation accuracy is relatively coarse.

There are 3 categtories should be noticed:
1. ExchangeAPI
2. Strategy
3.Backtest

Backtesting process:
1. Read OHLC data; 
2. Calculate indicators on OHLC data; 
3. The strategy decides buy or sell based on indicator vectors; 
4. Execute trades with the simulated "exchange";
5. Analyze the results.
'''

import pandas as pd

class Backtest:
    def __init__(self,
             data: pd.DataFrame,
             strategy_type: type,
             broker_type: type,
             cash: float = 10000,
             commission: float = 0.0
             ):
        data = data.copy()
        self._data = data
        self._broker = broker_type(data, cash, commission)
        self._strategy = strategy_type(self._broker, self._data)
        self._results = None
    
    def run(self):

        strategy = self._strategy
        broker = self._broker

        strategy.init()
        start = 100
        end = len(self._data)

        for i in range(start, end):
            broker.next(i)
            strategy.next(i)

        self.results = self._compute_result(broker)
        return self._results
    
    def _compute_result(self, broker):
        s = pd.Series()
        s['initial market value'] = broker.initial_cash
        s['ending_market_value'] = broker.ending_market_value
        s['profit'] = broker.market_value - broker.initial
        return s
    
    '''
    applyiing moving average Crossover Strategy
    '''

    def SMA(values n):
        return pd.Series(values).rolling(n).mean()
    
    def Crossover(series1, series2) -> bool:
        '''
        param series1: series1
        param series2: series2
        return: if crossover -> True, otherwise, Flase
        '''
        return series1[-3] < series2[-3] and series1[-2] > series2[-2] and series1[-1] > series2[-1]
    
    def next(self, tick):
        if crossover(self.sma1[:tick], self.sma2[:tick]):
            self.buy()

        elif crossover(self.sma2[:tick], self.sma1[:tick]):
            self.sell()

        else:
            pass
    
    

    

    


