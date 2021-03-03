import os
import copy
import pandas as pd
import argparse
# from pandas_datareader import data as pdr
from common.commondir import CommonDir
from common.readwrite import ReadFile, WriteFile
from IPython.display import display
##################################################

def data_check(starts):     
    portfolio = pd.DataFrame()
    data_dir = os.path.join(CommonDir.sampled_us_dir,'stocks')
    for data in os.listdir(data_dir):
        if data.endswith('csv') and data.startswith(starts):
            ss_dp = os.path.join(data_dir,data)
            if os.path.exists(ss_dp):
                ss_df = pd.read_csv(ss_dp)
                
                if ss_df.shape[0] > 0:                    
                    ss_df = ss_df.tail(1)
                    ss_df['ticker'] = data
                    if ss_df.shape[0] > 0:
                        portfolio = portfolio.append(ss_df)
                else:
                    print ('EMPTY : {}'.format(ss_dp) )
                
            
    if portfolio.shape[0] > 0:
        display(portfolio[['ticker', 'Open']])


def train_check(starts):
    portfolio = pd.DataFrame()
    data_dir = os.path.join(CommonDir.train_output_dir)
    for trainer in os.listdir(data_dir):
        if trainer.startswith(starts):
            ss_dp = os.path.join(data_dir, trainer, 'optimize', 'final', 'strategy.csv')
            print(ss_dp)
            if os.path.exists(ss_dp):
                ss_df = pd.read_csv(ss_dp)
                
                if ss_df.shape[0] > 0:                    
                    portfolio = portfolio.append(ss_df)
                else:
                    print ('EMPTY : {}'.format(ss_dp) )
                
            
    if portfolio.shape[0] > 0:
        cns = ['ticker','ave_pnl', 'sharpe', 'win_rate','zc', 'tot_qty', 'max_pos', 'days' ]
        display(portfolio[cns])
                    
#### ========================================================================
def main():
    starts = 'B'
    data_check(starts)
    train_check(starts)
        
        
if __name__ == '__main__':
    main()