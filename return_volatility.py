import pandas as pd

class OHLCAnalyzer:
    def __init__(self, ohlc_data):
        self.ohlc_data = ohlc_data

    def calculate_return_feature(self, periods, forward=False):
        return_feature = self.ohlc_data['Close'].pct_change(periods)
        return return_feature.shift(-periods) if forward else return_feature

    def calculate_volatility_feature(self, window):
        daily_returns = self.ohlc_data['Close'].pct_change()
        return daily_returns.rolling(window=window).std() * (252 ** 0.5)

    def prepare_features(self):
        feature_data = pd.DataFrame(index=self.ohlc_data.index)

        for period in [1, 3, 5, 10]:
            feature_data[f'forward_return_{period}'] = self.calculate_return_feature(period, forward=True)
            feature_data[f'return_{period}'] = self.calculate_return_feature(period)

        for window in [5, 10, 20]:
            feature_data[f'volatility_{window}'] = self.calculate_volatility_feature(window)

        return feature_data