import os
import copy
import pandas as pd
import argparse
# from pandas_datareader import data as pdr
from common.commondir import CommonDir
from common.readwrite import ReadFile, WriteFile
from IPython.display import display
##################################################
# def writeUSStrategy(portfolio):
inst =  {
            "symbol": "A",
            "exchange": "SMART",
            "secType": "STK",
            "currency": "USD",
            "stgy_cfg": {
                "bison_coeff": [],
                "salmon": {
                    "s_edge": 1,
                    "f_edge": 1,
                    "c_edge": 1,
                    "b_edge": 1
                },
                "max_pos": 10,
                "max_loss": 10000,
                "order_size": 1,
                "pos": 0,
                "tick_size" : 0.01
            }
        }
                
#### ========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--min_sharpe', default=2.0)
    args = parser.parse_args()
    
    portfolio_fp = os.path.join(CommonDir.sim_output_dir, 'portfolio.USStocks.csv')
    portfolio_df = pd.read_csv(portfolio_fp)
    
    print (portfolio_df.columns.tolist())
    
    select_df = pd.DataFrame()
    instruments = []
    for name, gr in portfolio_df.groupby('ticker'): 
        tdf = gr.sort_values(by='sharpe', ascending=False)
        if tdf['sharpe'].iloc[0] > args.min_sharpe:
            ticker = copy.deepcopy(inst)
            ticker['symbol'] = name
            ticker['stgy_cfg']['salmon']['s_edge'] = tdf['s_edge'].iloc[0]
            ticker['stgy_cfg']['salmon']['f_edge'] = tdf['f_edge'].iloc[0]
            ticker['stgy_cfg']['salmon']['c_edge'] = tdf['c_edge'].iloc[0]
            ticker['stgy_cfg']['salmon']['b_edge'] = tdf['b_edge'].iloc[0]
            oos_re = {'ave_pnl': tdf['ave_pnl'].iloc[0],
                      'sharpe': tdf['sharpe'].iloc[0],
                      'win_rate': tdf['winrate'].iloc[0],
                      'price_level': tdf['ref_price'].iloc[0]
                    }
            ticker['oos'] = oos_re
            instruments.append(ticker)
            select_df = select_df.append(tdf.head(1))            
    
    portfolio_fp = os.path.join(CommonDir.common_dir, 'instruments_us.json')
    portfolio_json = ReadFile.read_json(portfolio_fp)
    portfolio_json['USStock']['instruments'] = instruments
    WriteFile.write_json(os.path.join(CommonDir.common_dir,'instruments_us.prod.json'), portfolio_json)
    display(select_df[['ticker', 'ave_pnl', 'winrate', 'sharpe','tick_size']].sort_values(by=['sharpe'], ascending=False))
        
        
if __name__ == '__main__':
    main()