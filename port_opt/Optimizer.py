'''
Created on Jan 29, 2019

@author: smin
'''
import cma

class EvolutionOptimizer(object):
    def __init__(self, initList, step_size, bounds):
        two_value_bounds = self.__build_two_value_bounds(bounds)
        initValue = self.__adjustInitValue(initList, bounds)
        self.es = cma.CMAEvolutionStrategy(initValue, step_size, {'bounds': two_value_bounds})
        
    def __build_two_value_bounds(self, bounds):
        low_bounds = []
        upper_bounds = []
        for item in bounds:
            low_bounds.append(item[0])
            upper_bounds.append(item[1])
        return [low_bounds, upper_bounds]

    def __adjustInitValue(self, values, bounds):
        initValues = [None] * len(values)
        for i, v in enumerate(values):
            print (v, bounds[i][0])
            if v < bounds[i][0]:
                initValues[i] = bounds[i][0]
            elif v >  bounds[i][1]:
                initValues[i] = bounds[i][1]
            else:
                initValues[i] = v
        return initValues

    def optimize(self, object_func):
        while not self.es.stop():
            X = self.es.ask()
            self.es.tell(X, [object_func(x) for x in X])
            self.es.disp()  # by default sparse, see option verb_disp

        print('termination by', self.es.stop())
        print('best f-value =', self.es.result[1])
        print('best solution =', self.es.result[0])
        print('potentially better solution xmean =', self.es.result[5])
        return list(self.es.result[0])
        