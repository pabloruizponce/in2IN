import sys
sys.path.append(sys.path[0]+r"/../../../")

import torch
import random
import argparse
import numpy as np

from in2in.utils.configs import get_config
from in2in.models.dualmdm import load_DualMDM_model
from in2in.utils.metrics import calculate_wasserstein
from in2in.evaluation.utils import EvaluatorModelWrapper, get_dataset_motion_loader, get_motion_loader_DualMDM

# Randon Seed Configuration
np.random.seed(0)
random.seed(0)

def calculate_individual_diversity(generated_motion_embeddings):
    s_int = generated_motion_embeddings[:num_times]
    s_ind = generated_motion_embeddings[num_times:]
    
    diff, corrs_1_to_2, corrs_2_to_1 = calculate_wasserstein(s_int, s_ind, max_iters=500, verbose=True)
    return diff


def evaluation():
    metrics = {
        'Individual Diversity': [],
    }

    for i in range(replication_times):
        print("Replication: ", i)

        individual_diversity = []
        eval_motion_loader = eval_motion_loader_getter()

        for bidx ,eval_data in enumerate(eval_motion_loader):

            generated_motions1, generated_motions2, motion1, motion2, motion_lens, text, text_individual1, text_individual2 = eval_data

            # This is needed in order to work the motion emeddings
            generated_motions1 = generated_motions1[0]
            generated_motions2 = generated_motions2[0]
            motion1 = motion1[0]
            motion2 = motion2[0]
            motion_lens = motion_lens[0]
            
            generated_motion_embeddings = eval_wrapper.get_motion_embeddings([
                "name",
                text,
                generated_motions1,
                generated_motions2,
                motion_lens,
                text_individual1,
                text_individual2
            ])

            motion_len = motion_lens[0].item()
            generated_motions1 = generated_motions1[:,:motion_len,:]
            generated_motions2 = generated_motions2[:,:motion_len,:]
            motion1 = motion1[:,:motion_len,:]
            motion2 = motion2[:,:motion_len,:]

            individual_diversity.append(calculate_individual_diversity(generated_motion_embeddings).cpu().numpy().tolist())

        metrics['Individual Diversity'].append(np.mean(individual_diversity))
        print("Individual Diversity: ", np.mean(individual_diversity))

    print("---- Final Metrics ----")
    print("Individual Diversity: ", np.mean(metrics['Individual Diversity']), ", std: ",np.std(metrics['Individual Diversity']))

if __name__ == '__main__':

    # Configuration values
    num_samples = 100
    num_times = 32
    replication_times = 5
    batch_size = 1

    # Create the parser
    parser = argparse.ArgumentParser(description="Argparse example with optional arguments")

    # Add optional arguments
    parser.add_argument('--model', type=str, required=True, help='Model Configuration file')
    parser.add_argument('--evaluator', type=str, required=True, help='Evaluator Configuration file')
    parser.add_argument('--device', type=int, default=0, help='GPU device id')

    # Parse the arguments
    args = parser.parse_args()

    # Loading configuration files
    data_cfg = get_config("configs/datasets.yaml").interhuman_test
    model_cfg = get_config(args.model)
    evalmodel_cfg = get_config(args.evaluator)

    # Cuda configuration
    device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)

    # Build and Load Model
    model = load_DualMDM_model(model_cfg)

    # Get Datasets and DataLoaders
    gt_loader, gt_dataset = get_dataset_motion_loader(data_cfg, batch_size, num_samples=num_samples)
    eval_motion_loader_getter = lambda: get_motion_loader_DualMDM(batch_size, model, gt_dataset, device, num_times)

    # Evaluator Model
    eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device)
    evaluation()
