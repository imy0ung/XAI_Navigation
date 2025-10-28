from eval.habitat_multi_evaluator import HabitatMultiEvaluator
from config import load_eval_config
#from eval configimport MONActor
from eval.habitat_evaluator import Result
import pickle

def main():
    import os
    # Load the evaluation configuration
    eval_config = load_eval_config()
    # Create the HabitatEvaluator object
    evaluator = HabitatMultiEvaluator(eval_config.EvalConf, None)
    # data = evaluator.read_results('/home/finn/active/MON/results_vlfm/', "opt_PL", os.path.join("./", 'vlfm.pkl'))
    # pickle_path = os.path.join("./", 'vlfm.pkl')
    data = evaluator.read_results('results_multi/', "s",  os.path.join("./", 'multi.pkl'))
    # pickle_path = os.path.join("./", 'multi_noGlasses.pkl')
    # with open(pickle_path, 'wb') as f:
    #     pickle.dump(data, f)
    # print(data[data['state'] == 2]['experiment'].tolist())
    # print(data[data['scene'] == 'hm3d/val/00880-Nfvxx8J5NCo/Nfvxx8J5NCo.basis.glb'][data['state'] == 5].index.tolist())
    # print(data[data['state'] == 5].index.tolist())
    # print(len(data[data['obj'] == 'bed'].index.tolist()))


if __name__ == "__main__":
    main()
