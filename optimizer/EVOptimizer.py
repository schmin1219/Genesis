'''
Created on Dec 6, 2020

@author: smin
'''
import sys
import numpy as np
import pandas as pd
import cma

# EVOptimizer(self.edges, self.signals, json_)
class EVOptimizer(object):

    def __init__(self, edges, signals, json_):
        self.edges = edges
        self.signals = signals
        self.json_ = json_
        len_strategy_params = len(edges.keys())
        self.minmax = {} 
        self.sigma_fac = 1.0
        self.ask_update = False
        bounds = [0, np.inf] # None
        sigma0 = json_['optimizer']['config']['sigma']
        self.max_loop = json_['optimizer']['config']['max_loop']
        pop_size = json_['optimizer']['config']['pop_size']
#         initList = len_strategy_params * [ json_['optimizer']['config']['salmon_init_point'] ]+ len(signals) * [ json_['optimizer']['config']['bison_init_point'] ]
        initList = len_strategy_params * [ 10.0]+ len(signals) * [ 1.0 ] 
        self.es = cma.CMAEvolutionStrategy(initList, sigma0, {'bounds':bounds, 'seed':20181210, 'popsize':pop_size, 'verbose':True})
        self.pop_size = pop_size
        self.sigma0 = sigma0
        self.original_sigma = sigma0
        print ('real max_loop = {}'.format(self.max_loop))
        self.df = pd.DataFrame()

    def __adjustInitValue(self, values, bounds):
        initValues = [None] * len(values)
        for i, v in enumerate(values):
            if bounds == None:
                initValues[i] = v
            else:
                if v < bounds[i][0]:
                    initValues[i] = bounds[i][0]
                elif v >  bounds[i][1]:
                    initValues[i] = bounds[i][1]
                else:
                    initValues[i] = v
        return initValues


    def __optimize(self, object_func):
        idx = 0
        while not self.es.stop():
            X = self.es.ask()
            self.es.tell(X, [object_func(x) for x in X])
            self.es.disp()  # by default sparse, see option verb_disp
            if idx > self.max_loop:
                break
            idx = idx + 1
        return self.es.result


    def optimize(self, object_func):
        self.__optimize(object_func)
        print('termination by', self.es.stop())
        print('best f-value =', self.es.result[1])
        print('best solution =', self.es.result[0])
        print('potentially better solution xmean =', self.es.result[5])
        return list(self.es.result[0])


    def run(self, yvar, xvars, df, remove_neg_weight):
        self.df = df
        self.xvars = xvars
        self.yvar = yvar
        self.remove_neg_weight = remove_neg_weight
        self.minmax = { _x: (df[_x].min(),df[_x].max()) for idx, _x in enumerate(xvars) }
        self.optimize(self.__default_ls_objfunc)


    def __default_ls_objfunc(self, weights):
        return np.sum(((self.df[self.yvar] - (self.df[self.xvars] * weights).sum(axis=1)) ** 2))


    def to_result_json_dict(self, columns_dict=None):
        lst = {}
        lst['Best_SSR'] = np.round(self.es.result[1],5)
        lst['weights'] = [{'x': self.xvars[idx] if columns_dict is None else columns_dict[self.xvars[idx]]
                           , 'coeff': float(x) if self.remove_neg_weight==False or (self.remove_neg_weight and float(x) > 0.0) else 0.0
                           , 'effective_stds': self.es.result[6][idx]
                           ,'min':np.round(self.minmax[self.xvars[idx]][0], 5)
                           ,'max':np.round(self.minmax[self.xvars[idx]][1], 5)
                           } for idx, x in enumerate(self.es.result[0])]
        lst['evals_best'] = self.es.result[2]
        lst['name'] = 'EvolutionStrategy'
        return lst

    def to_result_df(self):
        df = pd.DataFrame(columns=self.xvars + ['score'])
        df.loc[0] = [ x for idx, x in enumerate(self.es.result[0]) ] + [np.round(self.es.result[1],5)]  # this score is SSR not r square
        return df

    def to_print(self, columns_dict=None):
        print (sys.stderr, 'EvolutionStrategy output result')
        print (sys.stderr, 'Best_SSR\t: {:f}'.format(np.round(self.es.result[1],5)))
        print (sys.stderr, 'evals_best\t: {:f}'.format(self.es.result[2]))
        print (sys.stderr, 'parameter list')
        params = [[self.xvars[idx] if columns_dict is None else columns_dict[self.xvars[idx]]
                       , np.round(x,5) if self.remove_neg_weight==False or (self.remove_neg_weight and x > 0.0) else 0.0
                       , self.es.result[6][idx]] for idx, x in enumerate(self.es.result[0])]
        for param in params:
            print (sys.stderr, '\t{}: coeff {:f}, effective_stds {:f}'.format(param[0], param[1],param[2]))

    def maxloop(self):
        return self.max_loop

    def buildUniqueWeights(self, unit):
        self.sigma_fac = 1.
        pops = self.es.ask(self.pop_size * 3, None, self.sigma_fac)
        newWeights = []
        isBuilt = False
        for idx in range(100):
            pops = self.es.ask(self.pop_size * 3, None, self.sigma_fac)
            for weights in pops:
                candidate = [ v - v % unit for v in weights ]
                if candidate not in newWeights:
                    newWeights.append(candidate)

                if len(newWeights) == self.pop_size:
                    isBuilt = True
                    break
            if isBuilt:
                break

            self.sigma_fac = self.sigma_fac+ 0.025
        return newWeights

    def ask(self, is_first=False, unit=0.005):
        
        if self.ask_update:
            self.sigma_fac = self.sigma_fac * 1.10
        else:
            self.sigma_fac = 1.0

        if unit > 0:
            newWeights = self.buildUniqueWeights(unit)
        else:
            newWeights =  self.es.ask(self.pop_size, None);

        if is_first:
            init_weights = len(newWeights[0]) * [1.0]
            newWeights[0] = np.array(init_weights)
        return newWeights


    def tell(self, X, f_values, isMax):
        pos = [ v for v in f_values if v > 0 ]
        if len(pos) > 0:
            self.ask_update = True
        else:
            self.ask_update = False

        pheno_value =  np.asarray(f_values) * -1.0 if isMax else f_values
        return self.es.tell(X, pheno_value)

    def exitCondition(self):
        return self.es.stop()

    def result(self):
        return self.es.result

     