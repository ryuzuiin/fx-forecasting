import pandas as pd
import talib


class TAFeatures:
    def __init__(self, ohlc_data):
        self.ohlc_data = ohlc_data
        self.open = ohlc_data['Open']
        self.high = ohlc_data['High']
        self.low = ohlc_data['Low']
        self.close = ohlc_data['Close']

    def ADX(self,timeperiod=14):
        return talib.ADX(self.high, self.low, self.close, timeperiod)

    def ADXR(self,timeperiod=14):
        return talib.ADXR(self.high, self.low, self.close, timeperiod)

    def APO(self, fastperiod=12, slowperiod=26, matype=0):
        return talib.APO(self.close, fastperiod, slowperiod, matype)

    def AROONOSC(self,timeperiod=14):
        return talib.AROONOSC(self.high, self.low, timeperiod)

    def BOP(self):
        return talib.BOP(self.open, self.high, self.low, self.close)

    def CCI(self,timeperiod=14):
        return talib.CCI(self.high, self.low, self.close, timeperiod)

    def MACD(self,fastperiod=12, slowperiod=26, signalperiod=9):
        macd, macdsignal, macdhist = talib.MACD(self.close, fastperiod, slowperiod, signalperiod)
        return macd

    def MOM(self,timeperiod=10):
        return talib.MOM(self.close, timeperiod)

    def RSI(self, timeperiod=14):
        return talib.RSI(self.close, timeperiod=14)

    def ULTOSC(self, timeperiod1=7, timeperiod2=14, timeperiod3=28):
        return talib.ULTOSC(self.high, self.low, self.close, timeperiod1, timeperiod2, timeperiod3)

    def BBANDS(self, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
        return talib.BBANDS(self.close, timeperiod, nbdevup, nbdevdn, matype)

    def DEMA(self, timeperiod=30):
        return talib.DEMA(self.close, timeperiod)

    def EMA(self, timeperiod=30):
        return talib.EMA(self.close, timeperiod)

    def MA(self, timeperiod=30, matype=0):
        return talib.MA(self.close, timeperiod, matype)

    def NATR(self, timeperiod=14):
        return talib.NATR(self.high, self.low, self.close, timeperiod)

    def TRANGE(self):
        return talib.TRANGE(self.high, self.low, self.close)

    def get_all_indicators(self):

        print(pd.__version__)
        indicators_df = pd.DataFrame(index=self.ohlc_data.index)

        indicators_df['ADX'] = self.ADX()
        indicators_df['ADXR'] = self.ADXR()
        indicators_df['APO'] = self.APO()
        indicators_df['AROONOSC'] = self.AROONOSC()
        indicators_df['BOP'] = self.BOP()
        indicators_df['CCI'] = self.CCI()
        indicators_df['MACD'] = self.MACD()
        indicators_df['MOM'] = self.MOM()
        indicators_df['RSI'] = self.RSI()
        indicators_df['ULTOSC'] = self.ULTOSC()
        indicators_df['DEMA'] = self.DEMA()
        indicators_df['EMA'] = self.EMA()
        indicators_df['MA'] = self.MA()
        indicators_df['NATR'] = self.NATR()
        indicators_df['TRANGE'] = self.TRANGE()
        return indicators_df

    

    



    

    

    