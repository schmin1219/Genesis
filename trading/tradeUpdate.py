import os
import time
import argparse
import pandas as pd
from common.commondir import CommonDir
##################################################

                
#### ========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--trd_date', help='trading_date', default='2021-01-15')
    parser.add_argument('-j', '--pname', default='smin0')
    args = parser.parse_args()
    date = args.trd_date
    pname = args.pname
    trade_csv_fp = os.path.join(CommonDir.trading_dir,'{}.orderUpdate.csv'.format(args.trd_date.replace('-','')))
    if os.path.exists(trade_csv_fp) == False:
        return
    trade_df = pd.read_csv(trade_csv_fp)
    trade_df = trade_df.reset_index(drop=True)
    
    for _, row in trade_df.iterrows():
        if int(row['qty']) == 0:
            continue
        
        cmd = 'PYTHONPATH=/home/smin/eclipse-workspace/BuySideStgy python /home/smin/eclipse-workspace/BuySideStgy/trading/update_portfolio.py '
        args = '-n {} -t {} -b {} -q {} -p {} -d {}'.format(pname, row['ticker'], row['buy_sell'], int(row['qty']), int(row['price']), date)
        cmd = cmd + args
        os.system(cmd)
        print (cmd)
        time.sleep(2) # 2 second sleep 
    

if __name__ == '__main__':
    main()