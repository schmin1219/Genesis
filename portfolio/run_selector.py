import os
import copy
import pandas as pd
import argparse
# from pandas_datareader import data as pdr
from common.commondir import CommonDir
from common.readwrite import ReadFile, WriteFile
from IPython.display import display
##################################################
def writeUSStrategy(portfolio):
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
    
    portfolio_fp = os.path.join(CommonDir.common_dir, 'instruments_us.json')
    portfolio_json = ReadFile.read_json(portfolio_fp)
    instruments = []
    for _, row in portfolio.iterrows():
        ticker = copy.deepcopy(inst)
        ticker['symbol'] = row['ticker']
        ticker['stgy_cfg']['salmon']['s_edge'] = row['s_edge']
        ticker['stgy_cfg']['salmon']['f_edge'] = row['f_edge']
        ticker['stgy_cfg']['salmon']['c_edge'] = row['c_edge']
        ticker['stgy_cfg']['salmon']['b_edge'] = row['b_edge']
        train_re = {'ticker_id' : row['ticker_id'],
                    'ave_pnl': row['ave_pnl'],
                    'win_pnl': row['win_pnl'],
                    'loss_pnl': row['loss_pnl'],
                    'sharpe': row['sharpe'],
                    'win_rate': row['win_rate'],
                    'tot_qty': row['tot_qty'],
                    'max_pos': row['max_pos'],
                    'worst_dd': row['worst_dd'],
                    'zc':row['zc'],
                    'days': row['days'],
                    'price_level': row['price_level']
                    }
        ticker['train'] = train_re
        instruments.append(ticker)
    portfolio_json['USStock']['instruments'] = instruments
    WriteFile.write_json(os.path.join(CommonDir.common_dir,'instruments_us.prod.json'), portfolio_json)
     
                
#### ========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--max_qty', help='max_qty', default=250)
    parser.add_argument('-s', '--min_sharpe', default=2.0)
    parser.add_argument('-p', '--max_price_level', default=600)
    parser.add_argument('-r', '--min_pnl_ratio', default=0.00)
    parser.add_argument('-w', '--min_winrate', default=0.7)
    parser.add_argument('-d', '--min_days', default = 50)
    args = parser.parse_args()
    
    cns = ['ave_pnl', 'sharpe', 'tot_qty','max_pos', 'c_edge', 'ticker', 'price_level','profit_ratio']
    portfolio = pd.DataFrame()
    for trainer in os.listdir(CommonDir.train_output_dir):
        fp = os.path.join(CommonDir.train_output_dir, trainer, 'optimize', 'final', 'historical_optimization.csv')
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            df =df [df.days > args.min_days]
#             df =df [df.tot_qty < args.max_qty]        
            df =df [df.win_rate> args.min_winrate]
            df =df [df.sharpe > args.min_sharpe]
            
            try:
                ss_dp = os.path.join(CommonDir.sampled_us_dir,'stocks', '{}USUSD.csv'.format(trainer))
                if os.path.exists(ss_dp):
                    ss_df = pd.read_csv(ss_dp)                    
                    ss_df = ss_df.tail(1)
                    last_price = ss_df['Close'].iloc[0]
                    if last_price > args.max_price_level:
                        print ('{} : {}'.format(trainer, last_price))
                        continue
            except KeyError:
                print ('{} failed to get value '.format(trainer))
                continue
            
            # Sorting
            df['profit_ratio'] = df['ave_pnl'] / last_price
            df = df.sort_values(by=['profit_ratio', 'ave_pnl', 'tot_qty', 'worst_dd' ], ascending=False)
            df = df.sort_values(by=['max_pos'], ascending=True)
            df['price_level'] = last_price            
            if df.shape[0] > 0:
                portfolio = portfolio.append(df.head(1))
    
    if portfolio.shape[0] > 0:
        portfolio = portfolio.sort_values(by=['profit_ratio', 'max_pos'],ascending=False).reset_index(drop=True)
        display(portfolio[cns].round(decimals=2))
        writeUSStrategy(portfolio)
        
        
if __name__ == '__main__':
    main()