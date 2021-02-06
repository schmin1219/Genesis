import json
import argparse
from pandas_datareader import data as pdr
from common.InstMaster import InstMaster
from IPython.display import display
##################################################

                
#### ========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--max_allocation', help='max allocation', default=10000000)
    parser.add_argument('-j', '--pname', default='smin0')
    parser.add_argument('--tag', default='SM0_21')
    args = parser.parse_args()

    date = '2021-01-07'
    max_allocation = args.max_allocation
    pname = args.pname
    instMaster = InstMaster()
    inst_json = instMaster.getInstrumentJson()
    
    inst_update = []
    for idx, inst in enumerate(inst_json[pname]):
        print (inst['symbol'])
        ss_df = pdr.get_data_yahoo(inst['symbol'], start=date, end=date)
        ss_df['d'] = ss_df.index
        ss_df = ss_df[ss_df['d'] == date]
        lprice = ss_df['Close'].iloc[0]
        max_alloc = int(max_allocation / lprice)
        inst['bounds'] = [1, max_alloc]
        inst_update.append(inst)
    
    inst_json[pname] = inst_update
    print (json.dumps(inst_json, indent=2))
    instMaster.writePortfolioJson(pname, inst_json)
        
if __name__ == '__main__':
    main()