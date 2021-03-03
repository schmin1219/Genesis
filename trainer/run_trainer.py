import os
import argparse
import time
import copy
import multiprocessing
from common.readwrite import ReadFile 
from common.commondir import CommonDir
from common.InstMaster import InstMaster
from optimizer.Optimizer import Optimizer


##################################################
def run_simOptimizer(data_dir,output_dir,tradingPeriod,symbol,sim_type,begin_date, end_date, obj_func, json_, intra_trading):
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)
    
    print ('CONTROL, Start task, run_simOptimizer {}'.format(symbol))
    start = time.time() 
    simOpt = Optimizer(data_dir, output_dir, tradingPeriod, symbol, sim_type, begin_date, end_date, obj_func, json_, intra_trading)
    simOpt.optimize()
    end = time.time()
    print ('CONTROL, Finished task, {} run_simOptimizer, {}'.format(symbol, end-start))
    return True

                
#### ========================================================================
def main():
    fp = os.path.join(CommonDir.sampled_us_dir, 'AUSUSD.csv')
    df = pd.read_csv(fp)
    print (df)
    
def main2():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--train_config', help='training config json', default='/home/smin/eclipse-workspace/IntraStrategy/models/US_model.json')
    args = parser.parse_args()
    
    if os.path.exists(args.train_config) == False:
        print ('{} does not exist'.format(args.train_config))
        return
    
    json_ = ReadFile.read_json(args.train_config)
    tradingPeriod   = int(json_['period'])
    sim_type        = int(json_['sim_type'])
    begin_date      = json_['begin_date']
    end_date        = json_['end_date']
    intra_trading = not json_['daily']
    train_output_dir =  CommonDir.train_output_dir
    if sim_type == 1:
        train_output_dir = CommonDir.train2_output_dir
    
    train_output_dir = os.path.join(train_output_dir, json_['output_dir'])
    instMaster = InstMaster(json_['daily'], json_['instruments'])
    cnt = multiprocessing.cpu_count() - 3
    symbols = instMaster.getInstNames()
    
    symbols = ['testdata']              #TODO testing 
    sampled_dir = '/home/smin/dataDir'  #TODO sampling directory
    
    for s in symbols:
        run_simOptimizer(os.path.join(sampled_dir, s)
                      ,os.path.join(train_output_dir, '{}'.format(s))
                      ,tradingPeriod, s, sim_type, begin_date, end_date, json_['obj_func']
                      ,copy.deepcopy(json_)
                      ,intra_trading)
    print ('end of optimization')



if __name__ == '__main__':
    main()