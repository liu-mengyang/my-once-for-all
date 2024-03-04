import time
import random
import argparse

import torch
import numpy as np

from ofa.model_zoo import ofa_net
from ofa.tutorial import AccuracyPredictor, FLOPsTable, LatencyProfiler, EvolutionFinder

# set random seed
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)



def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('weight_name', type=str)
    arg_parser.add_argument('backend_cfg', type=str)
    arg_parser.add_argument('command_cfg', type=str)
    
    args = arg_parser.parse_args()
    return args


def main():
    args = parse_args()
    
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(random_seed)
        print('Using GPU.')
    else:
        print('Using CPU.')
    
    ofa_network = ofa_net(args.weight_name, pretrained=True)
    print('The OFA Network is ready.')
    
    # accuracy predictor
    accuracy_predictor = AccuracyPredictor(
        pretrained=True,
        device='cuda:0' if cuda_available else 'cpu'
    )
    print('The accuracy predictor is ready!')
    
    latency_predictor = LatencyProfiler(
        args.backend_cfg,
        args.command_cfg
    )
    print('The latency profiler is ready!')
    
    
    latency_constraint = 40  # ms, suggested range [15, 33] ms
    P = 100  # The size of population in each generation
    N = 500  # How many generations of population to be searched
    r = 0.25  # The ratio of networks that are used as parents for next generation
    params = {
        'constraint_type': 'note10', # Let's do FLOPs-constrained search
        'efficiency_constraint': latency_constraint,
        'mutate_prob': 0.1, # The probability of mutation in evolutionary search
        'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
        'efficiency_predictor': latency_predictor, # To use a predefined efficiency predictor.
        'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
        'population_size': P,
        'max_time_budget': N,
        'parent_ratio': r,
        'ofa_network': ofa_network
    }

    # build the evolution finder
    finder = EvolutionFinder(**params)

    sample = finder.arch_manager.random_sample()
    ofa_network.set_active_subnet(ks=sample['ks'], d=sample['d'], e=sample['e'])
    model = ofa_network.get_active_subnet()
    efficiency = latency_predictor.predict_efficiency(model, [1,3,sample["r"][0],sample["r"][0]])

    # start searching
    result_lis = []
    st = time.time()
    best_valids, best_info = finder.run_evolution_search()
    result_lis.append(best_info)
    ed = time.time()
    search_time = ed-st
    print('Found best architecture on %s with latency <= %.2f ms in %.2f seconds! '
        'It achieves %.2f%s predicted accuracy with %.2f ms latency on %s.' %
        (target_hardware, latency_constraint, ed-st, best_info[0] * 100, '%', best_info[-1], target_hardware))

    # visualize the architecture of the searched sub-net
    _, net_config, latency = best_info
    ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
    print('Architecture of the searched sub-net:')
    print(ofa_network.module_str)

    print('-----------------------------------------')
    print("Time summary:")
    print("Convert: "+str(latency_predictor.convert_total_time))
    print("Send: "+str(latency_predictor.send_total_time))
    print("Profile: "+str(latency_predictor.profile_total_time))
    print("Total searching: "+str(search_time))
    
    
    
if __name__ == "__main__":
    main()
