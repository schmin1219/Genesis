import os
import argparse
import pandas as pd
import numpy as np
from common.commondir import CommonDir
from pandas_datareader import data as pdr
from common.InstMaster import InstMaster
from IPython.display import display
##################################################

                
#### ========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--trd_date', help='trading_date', default='2021-01-15')
    parser.add_argument('-j', '--pname', default='smin0')
    parser.add_argument('--tag', default='SM0_21')
    args = parser.parse_args()

    date = args.trd_date
    pname = args.pname
    instMaster = InstMaster()
    inst_json = instMaster.getInstrumentJson()
    portfolio_config_dir = os.path.join(CommonDir.train2_output_dir,args.tag)
    
    today_orders = pd.DataFrame(columns=['date','type', 'ticker', 'ticker_id', 'is_buy', 'price', 'qty', 'start_price'])
    
    for inst in inst_json[pname]:
        ticker = inst['canonical_name']
        ss_df = pdr.get_data_yahoo(inst['symbol'], start=date, end=date)
        ss_df['d'] = ss_df.index
        ss_df = ss_df[ss_df['d'] == date]
        config_fp = os.path.join(portfolio_config_dir, ticker, 'optimize', 'final', 'strategy.csv')
        if os.path.exists(config_fp) == False:
            continue
        config_df = pd.read_csv(config_fp)
        
        c_edge = config_df['c_edge'].iloc[0]
        s_edge = config_df['s_edge'].iloc[0]
        f_edge = config_df['f_edge'].iloc[0]
        pos = inst['pos'] if 'pos' in inst else 0
        orders = inst['orders'] if 'orders' in inst else []
        realized_pnl = inst['pnl'] if 'pnl' in inst else 0
        order_size = inst['order_size'] if 'order_size' in inst else 1  
        lprice = ss_df['Close'].iloc[0]
        
        if pos == 0:
            propBid = lprice - s_edge * instMaster.getTickSize(lprice)
            propAsk = propBid + c_edge * instMaster.getTickSize(lprice)
            today_orders.loc[today_orders.shape[0]] = [date, 'normal', inst['symbol'], inst['canonical_name'], True, propBid, order_size, lprice]
            today_orders.loc[today_orders.shape[0]] = [date, 'stop', inst['symbol'], inst['canonical_name'], False, propAsk, order_size, lprice]
        else:
            cov_notion = 0
            cov_qty = 0
            low_price = np.NAN
            for o in inst['orders']:
                cov_qty = cov_qty + o['qty']
                cov_notion = cov_notion + o['qty'] * o['price']
                if np.isnan(low_price) or low_price > o['price']:
                    low_price = o['price']
                  
            propAsk = cov_notion / cov_qty + c_edge * instMaster.getTickSize(lprice)
            buy_price = low_price - f_edge *  instMaster.getTickSize(lprice)
            propBid = lprice if buy_price > lprice else buy_price
            today_orders.loc[today_orders.shape[0]] = [date, 'normal', inst['symbol'], inst['canonical_name'], True, propBid, inst['order_size'], lprice]
            today_orders.loc[today_orders.shape[0]] = [date, 'stop', inst['symbol'], inst['canonical_name'], False, propAsk, o['qty'], lprice]
    
    display(today_orders)
    today_orders.to_csv(os.path.join(CommonDir.trading_dir, '{}.{}.csv'.format(pname,date)))

if __name__ == '__main__':
    main()