'''
Created on Jan 14, 2019

@author: smin
'''
import os
import json
import pandas as pd
import numpy as np
from skopt import gp_minimize
from common.InstMaster import InstMaster
import port_opt.Optimizer as eo
from common.commondir import CommonDir
from IPython.display import display


class PortfolioOptimizer(object):
    objfuncDict = {
                'maxSharpe'     :'neg_sharpe_ratio_objfunc',
                'minVolatility' :'min_volatility_objfunc'
    }

    algofuncDic = { 
                    'ES'        :'evolution_optimize'
    }

    def __init__(self, pname, trading_period):
        self.df = pd.DataFrame()
        self.trading_period = trading_period
        self.annualized_multiplier = 252 / self.trading_period
        self.instMaster = InstMaster()
        self.port_json = self.instMaster.getInstrumentJson()[pname]
        self.periods = self.getPeriods()
        for inst in self.port_json:
            tname = inst['canonical_name']
            self.df[tname] = self.getSimResults(tname)
        self.returns = self.df.replace('nan', np.nan).fillna(0)
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        self.risk_free_rate = 0.0
        self.results = []
        self.output_df = pd.DataFrame()
        self.symbol_list = []
        self.canonical_list = []
        
        
        bounds_lst = []
        for inst in self.port_json:
            bounds_lst.append(inst['bounds'])
            self.symbol_list.append(inst['symbol'])
            self.canonical_list.append(inst['canonical_name'])
        self.bounds = tuple (bounds_lst)


    def getPeriods(self):
        periods = []
        for fn in os.listdir(CommonDir.sim_output_dir):
            if fn.startswith('sim_') and fn.endswith('.csv') and 'instruments' not in fn:
                tokens = fn.replace('.','_').split('_')
                p = [tokens[2], tokens[3]]
                if p not in periods:
                    periods.append(p)

        print ('periods {}'.format(periods))        
        return periods 
         

    def getSimResults(self, tname):
        pnls = [] 
        sim_dir = CommonDir.sim_output_dir

        for period in self.periods:
            fp = os.path.join(sim_dir, 'sim_{}_{}_{}.csv'.format(tname, period[0], period[1]))
            if os.path.exists(fp):
                df = pd.read_csv(fp)
                if df.shape[0] > 0 :
                    pnls.append(df['ave_pnl'].mean())
                else:
                    pnls.append(np.NAN)
            else:
                pnls.append(np.NAN)
        return pnls

    def all_optimizing(self):
        for alg in self.algofuncDic: 
            for objfunc in self.objfuncDict:
                print ('algorithm ' , alg)
                self.optimize_specific_with_ind_bounds(alg, objfunc)


    def optimize_specific_with_ind_bounds(self, algo, obj_func):
        weights = eval('self.%s(algo,self.%s, \'%s\')'%(self.algofuncDic[algo], self.objfuncDict[obj_func], obj_func))
        sdp, rp, new_weights = self.__converted_output(weights)
        
        print ('new_weights', len(new_weights))
        
        #build up output
        max_sharpe_allocation = pd.DataFrame(new_weights,index=self.symbol_list,columns=['allocation'])
        
        
        max_sharpe_weights = max_sharpe_allocation.allocation 
        max_sharpe_allocation = max_sharpe_allocation.T
        output_name = '%s.%s' %(algo, obj_func)
        output = (rp, sdp, rp/sdp, self.symbol_list, max_sharpe_weights,output_name, self.canonical_list)
        self.results.append(output)

    
    # Objective function --------------------------------------------------------------------------
    def neg_sharpe_ratio_objfunc(self, weights):
        stdev, profit = self.__portfolio_annualised_performance(weights, self.mean_returns, self.cov_matrix)
        return -(profit - self.risk_free_rate) / stdev

    def min_volatility_objfunc(self, weights):
        return self.__portfolio_annualised_performance(weights, self.mean_returns, self.cov_matrix)[0]


    # Private member--------------------------
    def __portfolio_volatility(self, weights, mean_returns, cov_matrix):
        return self.__portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]


    #TODO this one should be revised 
    def __portfolio_annualised_performance(self, weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns*weights ) * self.annualized_multiplier
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(self.annualized_multiplier)
        return std, returns

    # ----------------- Optimizer Algorithm -----------------
    def evolution_optimize(self, algo, obj_func, f_name):
        num_assets = len(self.mean_returns)
        evol_optim = eo.EvolutionOptimizer(num_assets*[5], 2.0, self.bounds)
        weights = evol_optim.optimize(obj_func)
        return weights
    
    def baysian_optimize(self, algo, obj_func, f_name):
        num_assets = len(self.mean_returns)
        if f_name == 'maxSharpe':        
            results = gp_minimize(self.neg_sharpe_ratio_objfunc, self.bounds, x0=num_assets*[1./num_assets,], noise="gaussian")
        elif f_name == 'minVolatility':
            results = gp_minimize(self.min_volatility_objfunc, self.bounds, x0=num_assets*[1./num_assets,], noise="gaussian")
        return results.x

    
    def flatweight_optimize(self, algo, obj_func, bounds,f_name):
        return [1./len(self.mean_returns)]* len(self.mean_returns)
    
    ## ---------------------------------------------------------------------
 

    def __converted_output(self, weights):
        weights = np.round(weights,0)
        returns = np.sum(self.mean_returns*weights ) *  self.annualized_multiplier
        std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(self.annualized_multiplier)
        return std, returns, weights


    def buildResultTable(self):
        if self.output_df.empty:
            symbols = self.symbol_list
            symbols.append('annulaized pnl')
            symbols.append('annulaized volatility')
            symbols.append('annulaized sharpe')
            self.output_df['symbol'] = symbols            

            canonical_lst = self.canonical_list
            canonical_lst.append('NA')
            canonical_lst.append('NA')
            canonical_lst.append('NA')
            self.output_df['canonical_name'] = canonical_lst 

            #output = (rp, sdp, rp/sdp, self.symbol_list, max_sharpe_weights,output_name, self.canonical_name)
            for output in self.results:
                lst = []
                lst = list(np.round(output[4],2))
                lst.append(np.round(output[0],2))   # pnl
                lst.append(np.round(output[1],2))   # volatility
                lst.append(np.round(output[2],2))   # sharpe
                self.output_df[output[5]] = lst
        return self.output_df
        


######################### BEGIN MAIN() ########################
def test():
    portfolioOptimizer = PortfolioOptimizer('smin0', 126)
    portfolioOptimizer.optimize_specific_with_ind_bounds('ES', 'maxSharpe')
    df = portfolioOptimizer.buildResultTable()
    display(df)
    


if __name__ == "__main__":
    test()
