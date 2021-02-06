'''
Created on Dec 25, 2020

@author: smin
'''
import os
import argparse
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from pandas_datareader import data as pdr
from common.InstMaster import InstMaster
from common.commondir import CommonDir

from IPython.display import display


##################################################

                
#### ========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', help='intra, inter', default='inter')
    parser.add_argument('-i', '--intruments', help='instruments_kospi200, instruments_sp500', default='instruments_kospi200')
    args = parser.parse_args()

    output_dir = '/home/smin/StockEngine/data'
    start_date = '2016-01-01'
    end_date = '2020-12-25'    
    instMaster = InstMaster(args.type, args.instruments)
    inst_json = instMaster.getInstrumentJson()
    
    if instMaster.type() == 0:
        for inst in inst_json['instruments']:
            ss_df = pdr.get_data_yahoo(inst['symbol'], start=start_date, end=end_date)
            fp = '{}.csv'.format(os.path.join(output_dir, inst['canonical_name']))
            ss_df.to_csv(fp) 
            print (fp)
    else:
        for inst in inst_json['instruments']:
            symbol_ = inst['symbol']
            ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
            data, meta_data = ts.get_intraday(symbol=symbol_,interval='1min', outputsize='full')
            os.makedirs(os.path.join(CommonDir.sampled_us_dir, '{}'.format(symbol_)))
                        
            data.to_csv( symbol and date .csv )

if __name__ == '__main__':
    main()