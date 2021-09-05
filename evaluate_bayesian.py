import torch

import utils
import os
from tqdm.auto import tqdm
import numpy as np
from dataloaders import loader
from torch.nn import functional as F

from models.BayesBeat import Bayesian_Deepbeat

SAVE_PATH = 'saved_model/bayesbeat_cpu.pt'
ENSEMBLES = 1
DEEPBEAT_DATA_PATH = "data/"


def get_uncertainty(model, input_signal, T=15, normalized=False):
    """
    Batchified version
    """
    # [b, 1, 800] -> [T*b, 1, 800]
    input_signals = torch.repeat_interleave(input_signal, T, dim=0)

    # [T*b, 2]
    net_out, _ = model(input_signals)

    if normalized:
        # [T*b, 2] -> [b, T, 2]
        prediction = F.softplus(net_out).view(-1, T, 2)
        # [b, T, 2] / ([b, T, 2] -> [b, T] -> [b, T, 1]) -> [b, T, 2]
        p_hat = prediction / torch.sum(prediction, dim=2).unsqueeze(2)
    else:
        # [T*b, 2] -> [b, T, 2]
        p_hat = F.softmax(net_out, dim=1).view(-1, T, 2)

    p_hat = p_hat.detach().cpu().numpy()

    # [b, T, 2] -> [b, 2]
    p_bar = np.mean(p_hat, axis=1)

    # [b, T, 2] - [b, 2] -> [b, T, 2]
    temp = p_hat - np.expand_dims(p_bar, 1)

    epistemics = np.zeros((temp.shape[0], 2))
    aleatorics = np.zeros((temp.shape[0], 2))

    "Need to vectorize this loop"
    for b in range(temp.shape[0]):
        # [, 2, T] * [, T, 2] -> [, 2, 2]
        epistemic = np.dot(temp[b].T, temp[b]) / T
        # [, 2, 2] -> [, 2]
        epistemics[b] = np.diag(epistemic)

        # [, 2, T] * [, T, 2] -> [, 2, 2]
        aleatoric = np.diag(p_bar[b]) - (np.dot(p_hat[b].T, p_hat[b]) / T)
        # [, 2, 2] -> [, 2]
        aleatorics[b] = np.diag(aleatoric)

    return torch.Tensor(p_bar).to(input_signal.device), epistemics, aleatorics


def evaluate_with_uncertainity(model, generator, prefix='val_', uncertainity_bound=0.05, device=torch.device('cpu')):
    # EVALUATE SIGNAL BY SIGNAL WITH UNCERTAINTY CALCULATION

    REPEAT = 15
    stat_map_overall = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    output_map_all = {}
    output_map_all['true'] = torch.empty((0, 2), device=device)
    output_map_all['pred'] = torch.empty((0, 2), device=device)
    nAccepted = 0
    nSamples = 0

    for i, batch in enumerate(tqdm(iter(generator))):
        signal, _, rhythm = batch
        signal, rhythm = signal.to(device, non_blocking=True), rhythm.to(device, non_blocking=True)

        nSamples += len(rhythm)

        log_outputs, ep, al = get_uncertainty(model, signal, REPEAT)
        if uncertainity_bound > 0:
            mask = al[:, 1] < uncertainity_bound
            log_outputs = log_outputs[mask]
            rhythm = rhythm[mask]
            if len(log_outputs) == 0:
                continue

        nAccepted += len(rhythm)
        stat_map_batch = utils.batch_stat_deepbeat(rhythm_true=rhythm, rhythm_pred=log_outputs)
        utils.accumulate_stat(stat_map=stat_map_overall, **stat_map_batch)
        utils.accumulate_responses(output_map_all, rhythm, log_outputs)

    metrics_map = utils.metrics_from_stat(**stat_map_overall, prefix=prefix, output_map_all=output_map_all)
    metrics_map["coverage"] = nAccepted / float(nSamples)
    utils.print_stat_map(metrics_map)
    return metrics_map[prefix + 'F1']


if __name__ == '__main__':
    model = Bayesian_Deepbeat()

    device = 'cuda:0'
    print("Device: " + device)

    # BATCH_SIZE SHOULD BE 1 IN CURRENT BUILD. FULL MINIBATCH INFERENCE WILL BE ADDED IN FUTURE.

    test_generator = loader.get_weighted_generator(DEEPBEAT_DATA_PATH + 'test',
                                                   batch_size=1,
                                                   replacement=False, is_train=False, shuffle=False, remove_poor=False)

    best_val_f1_pos = -1

    model.load_state_dict(torch.load(os.path.join(SAVE_PATH))['state_dict'])
    model = model.to(device)
    model.eval()
    bounds = [0.05]
    with torch.no_grad():
        for bound in bounds:
            print('**************BOUND=: ' + str(bound) + '*************')
            print("-------------------------------------------------------")
            test_f1_pos = evaluate_with_uncertainity(model=model, generator=test_generator, prefix='test_',
                                                         uncertainity_bound=bound)
