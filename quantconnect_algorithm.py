# region imports
from AlgorithmImports import *
# endregion

class AssetAllocationWithMPBollinger(QCAlgorithm):

    def Initialize(self):
        # 1. Backtest Period and Cash
        self.SetStartDate(2023, 1, 1)  
        self.SetEndDate(2025, 12, 31)    
        self.SetCash(100000)           

        # 2. Portfolio Allocation Targets (Total 99%)
        # These weights are applied to the TOTAL current Net Liquidation Value (AUM)
        self.targets = {
            "VTI": 0.55,  # Total Stock Market
            "GLD": 0.03,  # Gold
            "REMX": 0.02, # Rare Earths ETF
            "BND": 0.39   # Total Bond Market
        }
        
        # 3. Active Asset (1% for MP Materials)
        self.mp_symbol = self.AddEquity("MP", Resolution.Daily).Symbol
        self.active_target = 0.01

        # Add Equities for static assets
        self.symbols = {}
        for ticker in self.targets:
            self.symbols[ticker] = self.AddEquity(ticker, Resolution.Daily).Symbol

        # 4. Bollinger Bands for MP Materials
        self.bb = self.BB(self.mp_symbol, 20, 2, MovingAverageType.Simple, Resolution.Daily)
        
        # Warm up indicators
        self.SetWarmUp(20)

        # 5. Dynamic Rebalancing
        # Rebalance the static portfolio on the first of every month 
        # to ensure weights stay correct as AUM grows/shrinks.
        self.Schedule.On(self.DateRules.MonthStart("VTI"), \
                         self.TimeRules.AfterMarketOpen("VTI", 30), \
                         self.Rebalance)

    def OnData(self, data: Slice):
        if self.IsWarmingUp or not self.bb.IsReady or not data.ContainsKey(self.mp_symbol): 
            return

        # Bollinger Band Trading Logic for MP Materials
        # SetHoldings(symbol, 0.01) automatically calculates the position size
        # based on the CURRENT Total Portfolio Value (AUM).
        price = self.Securities[self.mp_symbol].Price
        
        # BUY: Price below Lower Band
        if price < self.bb.LowerBand.Current.Value:
            if not self.Portfolio[self.mp_symbol].Invested:
                # This buys 1% of current AUM
                self.SetHoldings(self.mp_symbol, self.active_target)
                self.Debug(f"[{self.Time}] MP Buying @ {price}. AUM: {self.Portfolio.TotalPortfolioValue}")
        
        # SELL: Price above Upper Band
        elif price > self.bb.UpperBand.Current.Value:
            if self.Portfolio[self.mp_symbol].Invested:
                self.Liquidate(self.mp_symbol)
                self.Debug(f"[{self.Time}] MP Liquidating @ {price}")

    def Rebalance(self):
        # As Assets Under Management (AUM) increase, SetHoldings will 
        # scale the positions to maintain the exact percentage of the NEW total value.
        self.Debug(f"[{self.Time}] Monthly Rebalance. Current AUM: {self.Portfolio.TotalPortfolioValue}")
        for ticker, weight in self.targets.items():
            # This adjusts each holding to be exactly its % of the current AUM
            self.SetHoldings(self.symbols[ticker], weight)
