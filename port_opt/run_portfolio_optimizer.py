import os
import sys
import argparse
from common.commondir import CommonDir 
from port_opt.main_optimizer import PortfolioOptimizer
from IPython.display import display


usages=' Usage:\n    \
    Using Json binding file to run indv optimize :run_portfolio_optimizer.py -d [BASE_DIR] -b [BINDING_JSON_FILE] -o [OPTIMIZATION_TYPE] -t [OBJ_FUNC] --output_dir [OUTPUT_DIR]\n    \
    Using default binding value to run indv optimize : run_portfolio_optimizer.py -d [BASE_DIR] -u [Default UpperBound int] -o [OPTIMIZATION_TYPE] -t [OBJ_FUNC] --output_dir [OUTPUT_DIR] --has_min_one_lot\n    \
    Using Json binding file to run all optimize : run_portfolio_optimizer.py -d [BASE_DIR] -u [BINDING_JSON_FILE] -o all --output_dir [OUTPUT_DIR] --has_min_one_lot\n    \
    Using default binding value to run indv optimize : run_portfolio_optimizer.py -d [BASE_DIR] -u [Default UpperBound int] -o all --output_dir [OUTPUT_DIR] --has_min_one_lot\n    \
    Generate bounding json : run_portfolio_optimizer.py -d [BASE_DIR] -u [Default UpperBound int] --output_dir [OUTPUT_DIR] --gen_bound_json --has_min_one_lot\n    \
    '

def __help():
    print >> sys.stderr, ' mandatory fields are missed '
    print >> sys.stderr, usages


def main():
    parser = argparse.ArgumentParser(usage=usages)
    parser.add_argument('-p', '--pname', help='portfolio name', default='smin1')
    parser.add_argument('-t', '--trading_period', help='trading_period', default='126')
    args = parser.parse_args()
 
    portfolioOptimizer = PortfolioOptimizer(args.pname, int(args.trading_period))
    portfolioOptimizer.all_optimizing()
    
    df = portfolioOptimizer.buildResultTable()
    display(df)
    df.to_csv(os.path.join(CommonDir.portfolio_dir, '{}.optimization.csv'.format(args.pname)))

if __name__ == "__main__":
    main()
