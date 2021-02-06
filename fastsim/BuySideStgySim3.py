'''
Created on Dec 1, 2020

@author: smin
'''
import copy
import pandas as pd
import numpy as np
from fastsim.Order import MyOrder  

from IPython.display import display

'''
    This strategy is about intra-day strategy.
     
'''

class BuySideStgySim3(object):

    def __init__(self, df_, sedge, fedge, cedge, bedge, ticker, ticker_id, max_pos, order_size, max_loss):
        self.df = copy.deepcopy(df_)
        self.ticker = ticker
        self.ticker_id = ticker_id
        self.s_edge = sedge
        self.f_edge = fedge
        self.c_edge = cedge 
        self.b_edge = bedge
        self.position = 0
        self.columns = ['Date', 'Open', 'Close', 'Low', 'High']
        self.max_pos = max_pos
        self.order_size = order_size
        self.max_loss = max_loss
        self.order_id = 0
        self.orders = []
        self.pnls = []
        self.orderId = 0
        self.ord_cnt = 0 
        self.result = {
            'totalInvValue' :0
            ,'totalTradeQty':0
            ,'minUnRealizedPnL':0
            ,'maxUnRealizedPnL':0
            ,'maxTradedPosition':0
            }
        self.mid_cn = 'MidPriceSignalZC0'
        self.min_cn = 'MidPriceSignalZC0'
        self.max_cn = 'MidPriceSignalZC0'
        

    def get_bought(self, order, min_price, max_price):
        self.orderId = self.orderId + 1
        self.position = self.position + self.order_size
        self.orders.append(order)
        self.orders.sort(key=lambda order:order.date, reverse=False)
        self.result['totalInvValue']=self.result['totalInvValue'] + order.price * order.order_qty
        self.result['totalTradeQty']=self.result['totalTradeQty'] + order.order_qty
        self.result['maxTradedPosition'] = self.result['maxTradedPosition'] if self.result['maxTradedPosition'] >= self.position else self.position
        self.result['minUnRealizedPnL'] = self.position * min_price - self.result['totalInvValue']
        self.result['maxUnRealizedPnL'] = self.position * max_price - self.result['totalInvValue']
        

    def get_sold(self, min_price, max_price, tick_size, day, is_close):
        sold_cnt = 0
        for idx, o in enumerate(self.orders):
            cov_price = o.price + self.c_edge * tick_size
            if max_price > cov_price and o.order_qty > 0 or is_close:
                price = max_price if max_price < cov_price else cov_price
                price = min_price if price < min_price else price 
                cover_order =  MyOrder(self.orderId, False, price, o.order_qty, day)
                
                self.result['totalInvValue']=self.result['totalInvValue'] - cover_order.price * cover_order.order_qty
                self.result['totalTradeQty']=self.result['totalTradeQty'] + cover_order.order_qty
                sell_qty = cover_order.order_qty
                self.position = self.position - o.order_qty
                
                pnl = (cover_order.price - o.price) * sell_qty                    
                self.pnls.append(pnl) 
                self.orders[idx].order_qty = 0
                sell_qty = 0
                self.result['minUnRealizedPnL'] = self.position * min_price - self.result['totalInvValue']
                self.result['maxUnRealizedPnL'] = self.position * max_price - self.result['totalInvValue']
                sold_cnt = sold_cnt+1
                
        self.orders = [ o for o in self.orders if o.order_qty > 0 ]
        ret = True if sold_cnt > 0 and len(self.orders) == 0 else False
        return ret
            
    
    def run(self):
        alpha = self.df['alpha'].iloc[0]
        mid_price = self.df[self.mid_cn].iloc[0]
        tick_size = 5  # 0.01 # 1 cent based simulation
        propBid = mid_price - self.s_edge * tick_size + alpha
        
        for _, row in self.df.iterrows():
            min_price = row[self.min_cn]
            max_price = row[self.max_cn]
            mid_price = row[self.mid_cn]
            alpha = row['alpha']
             
            if self.get_sold(min_price, max_price, tick_size, self.ticker_id, False):
                propBid = mid_price - self.s_edge * tick_size + alpha 
                
            if self.position == 0 and min_price <= propBid:
                qty = self.max_pos - self.position if self.max_pos - self.position < self.order_size else self.order_size
                order = MyOrder(self.orderId, True, propBid, qty, self.ticker_id)
                self.get_bought(order, min_price, max_price)
            elif self.position > 0:
                order = self.orders[-1]
                propBid = order.price - self.f_edge * tick_size
                if self.max_pos > self.position and min_price <= propBid:
                    qty = self.max_pos - self.position if self.max_pos - self.position < self.order_size else self.order_size
                    order = MyOrder(self.orderId, True, propBid, qty, self.ticker_id)
                    self.get_bought(order, min_price, max_price)
            
        
        if len(self.orders) > 0:
            ldf = self.df.tail(1)
            end_price = ldf[self.mid_cn].iloc[0]
            self.get_sold(min_price, end_price, tick_size, self.ticker_id, True)         
        

    def get_res(self):
        self.pnls = [ pnl for pnl in self.pnls if np.isnan(pnl) == False ]
        zc = len(self.pnls)
        cum_pnl = sum(self.pnls)
        apnl = np.nanmean(self.pnls) if zc > 0 else np.NAN
        wpnls = [pnl for pnl in self.pnls if pnl > 0]
        lpnls = [pnl for pnl in self.pnls if pnl < 0]
        wpnl = np.nanmean(wpnls) if len(wpnls) > 0 else 0
        lpnl = np.nanmean(lpnls) if len(lpnls) > 0 else 0
        pnl_std = np.nanstd(self.pnls)
        sharpe = apnl/pnl_std if pnl_std > 0 else np.NAN 
        winrate = float(len(wpnls))/float(len(self.pnls)) if len(self.pnls) > 0 else 0
        totalqty = self.result['totalTradeQty']
        res_columns = ['ticker_id', 'ave_pnl', 'win_pnl', 'loss_pnl', 'sharpe', 'win_rate', 'tot_qty', 's_edge', 'f_edge', 'c_edge', 'max_pos', 'worst_dd', 'max_unreal_pnl', 'zc', 'cum_pnl']
        rdf = pd.DataFrame(columns=res_columns)
        rdf.loc[0] = [self.ticker_id, apnl, wpnl, lpnl, sharpe, winrate, totalqty, self.s_edge, self.f_edge, self.c_edge, self.result['maxTradedPosition'], self.result['minUnRealizedPnL'], self.result['maxUnRealizedPnL'], zc, cum_pnl]
        return rdf
        
    def getTickSize(self, price):
        tick_size = 1000
        if price < 1000:
            tick_size = 1
        elif price >= 1000 and price < 5000:
            tick_size = 5
        elif price >= 5000 and price < 10000:
            tick_size = 10
        elif price >= 10000 and price < 50000:
            tick_size = 50
        elif price  >= 50000 and price < 100000:
            tick_size = 100
        elif price >= 100000 and price < 500000:
            tick_size = 500
        else:
            tick_size = 1000
        return tick_size
            
    
       
def main():
    print ('test')

if __name__ == '__main__':
    main()        