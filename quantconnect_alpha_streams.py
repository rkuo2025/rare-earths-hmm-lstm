# region imports
from AlgorithmImports import *
# endregion

class AlphaStreamsEligibleAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2025, 12, 31)
        self.SetCash(100000)

        # 1. Set Alpha Streams Brokerage Model
        self.SetBrokerageModel(BrokerageName.AlphaStreams)

        # 2. Universe Selection
        self.static_tickers = ["VTI", "BND", "GLD", "REMX"]
        self.mp_ticker = "MP"
        self.symbols = [self.AddEquity(ticker, Resolution.Daily).Symbol for ticker in self.static_tickers + [self.mp_ticker]]
        self.SetUniverseSelection(ManualUniverseSelectionModel(self.symbols))

        # 3. Alpha Model
        self.AddAlpha(RareEarthsAlphaModel(self.static_tickers, self.mp_ticker))

        # 4. Portfolio Construction
        # InsightWeightingPortfolioConstructionModel respects the 'weight' property on Insights
        self.SetPortfolioConstruction(InsightWeightingPortfolioConstructionModel())

        # 5. Execution & Risk
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

class RareEarthsAlphaModel(AlphaModel):
    def __init__(self, static_tickers, mp_ticker):
        self.static_tickers = static_tickers
        self.mp_ticker = mp_ticker
        self.static_weights = {
            "VTI": 0.55,
            "BND": 0.39,
            "GLD": 0.03,
            "REMX": 0.02
        }
        self.bb = {}
        self.mp_symbol = None
        self.mp_direction = InsightDirection.Flat

    def OnSecuritiesChanged(self, algorithm, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol.Value == self.mp_ticker:
                self.mp_symbol = symbol
                # Bollinger Bands (20, 2)
                self.bb[symbol] = algorithm.BB(symbol, 20, 2, MovingAverageType.Simple, Resolution.Daily)

    def Update(self, algorithm, data):
        insights = []

        # 1. Emit/Refresh Insights for Static Assets (99%)
        # We emit these daily to maintain the allocation as AUM grows/changes
        for ticker in self.static_tickers:
            symbol = [s for s in algorithm.ActiveSecurities.Keys if s.Value == ticker][0]
            weight = self.static_weights[ticker]
            insights.append(Insight.Price(symbol, timedelta(days=1), InsightDirection.Up, None, None, None, weight))

        # 2. Bollinger Band Logic for MP Materials (1%)
        if self.mp_symbol in data.Bars:
            price = data[self.mp_symbol].Close
            indicator = self.bb[self.mp_symbol]
            
            if not indicator.IsReady: return insights

            # BUY Signal (Below Lower Band)
            if price < indicator.LowerBand.Current.Value:
                self.mp_direction = InsightDirection.Up
            
            # SELL Signal (Above Upper Band)
            elif price > indicator.UpperBand.Current.Value:
                self.mp_direction = InsightDirection.Flat

            # Emit MP Insight with its 1% weight
            insights.append(Insight.Price(self.mp_symbol, timedelta(days=1), self.mp_direction, None, None, None, 0.01))

        return insights
