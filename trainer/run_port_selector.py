import os
import argparse
import time
import pandas as pd
from common.commondir import CommonDir
from common.InstMaster import InstMaster
from optimizer.Optimizer import Optimizer

##################################################

                
#### ========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sharpe', help='minimum sharpe', default=1)
    parser.add_argument('-t', '--train_result', default='SM0_21')
    parser.add_argument('-p', '--portfolio_name', default='smin1')
    args = parser.parse_args()
    
    sharpe = float(args.sharpe)
    train_result_dir = os.path.join(CommonDir.train_output_dir, args.train_result) 
    
    instMaster = InstMaster()
    json_ = instMaster.getInstrumentJson()
    selected_inst_json = []
    for inst in json_['instruments']:
        symbol = inst['canonical_name']
        
        fp = os.path.join(train_result_dir, symbol, 'optimize', 'final', 'strategy.csv' )
        if os.path.exists(fp) == False:
            continue
        
        stgy_df = pd.read_csv(fp)
        if stgy_df['sharpe'].iloc[0] >= sharpe:
            inst['pos'] = 0
            inst['order_size'] = 1
            inst['pnl'] = 0
            inst['bounds'] = [1, 30]
            selected_inst_json.append(inst)
            print (fp)

    if len(selected_inst_json) > 0 :    
        instMaster.writePortfolioJson(args.portfolio_name, selected_inst_json)
    
    

if __name__ == '__main__':
    main()