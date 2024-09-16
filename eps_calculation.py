
import os
import time
import numpy as np

import torch

from score_sde import sde_lib
from score_sde.models import utils as mutils

from utils import load_data_from_train

from runners.diffpure_sde import RevVPSDE


def get_eps(x, model, args, config):
    #1. 使用一个5000大小的natural样本subset，from cifar10，计算出一个平均/max的eps值，2. 之后所有的input x将于这个值进行比较
    # ngpus
    # if "imagenet" in args.data:
    #     #split x into batches
    #     pass
    # Initialize score models.
    start_time = time.time()
    if args.dataset =="cifar10":
        img_shape = (3, 32, 32)
    elif args.dataset =="imagenet":
        img_shape = (3, 256, 256)
    rev_vpsde = RevVPSDE(model=model, score_type=args.score_type, img_shape= img_shape, model_kwargs= None).to(config.device)
    sde = sde_lib.VPSDE(beta_min=rev_vpsde.beta_0, beta_max=rev_vpsde.beta_1, N=rev_vpsde.N)
    score_fn = mutils.get_score_fn(sde, rev_vpsde.model, train=False, continuous=True)
    score_sum = torch.zeros_like(x, device=x.device)
    score_adv_list = []
    score_adv_lists = []
    # z_list = []
    with torch.no_grad():
        for value in range(1,args.diffuse_t+1):
            t_value = value/1000
            curr_t_temp = torch.tensor(t_value,device=x.device)

            z = torch.randn_like(x, device=x.device)
            # z_list.append(z)
            mean_x_adv, std_x_adv = sde.marginal_prob(2*x-1, curr_t_temp.expand(x.shape[0])) #Here the input should be in size of [0,1], converted to [-1,1]
            perturbed_data = mean_x_adv + std_x_adv[:, None, None, None] * z
            score = score_fn(perturbed_data, curr_t_temp.expand(x.shape[0]))

            if args.dataset=='imagenet':
                score, _ = torch.split(score, score.shape[1]//2, dim=1)
                assert x.shape == score.shape, f'{x.shape}, {score.shape}'
            if args.detection_ensattack_norm_flag:
                score_adv_list.append(score.detach().view(x.shape[0],-1).norm(dim=-1).unsqueeze(0))
            elif args.single_vector_norm_flag:
                # score_adv_list.append(score.detach().view(1, *x_adv.shape))
                score_sum += score.detach()
            else:
                score_adv_list.append(score.detach())
            # print(f"{value}th eps calculation time: {time.time() - start_time}")

        if args.detection_ensattack_norm_flag:
            score_adv_lists.append(torch.cat(score_adv_list, dim=0))
        elif args.single_vector_norm_flag:
            score_adv_lists.append(score_sum/value)
        else:
            score_adv_lists.append(torch.cat(score_adv_list, dim=0).view(len(score_adv_list),*x.shape).cpu())
    
    print(f'diffuison time: {time.time() - start_time}')
    adv_score_tensor = torch.cat(score_adv_lists, dim=1) if not args.single_vector_norm_flag else torch.cat(score_adv_lists, dim=0)
    # print(f"adv_score_tensor.shape:{adv_score_tensor.shape}")
    
    # if not args.single_vector_norm_flag:
    #     # eps分数
    #     np.save(f"{args.eps_norm_dir}/{x.shape[0]}_adv_eps.npy", adv_score_tensor.data.cpu().numpy())
    # else:
    #     # eps vector
    #     np.save(f"{args.eps_norm_dir}/{x.shape[0]}_adv_eps_vector.npy", adv_score_tensor.data.cpu().numpy())
    return adv_score_tensor



def reweight_t(x_test, model, eps_data, args, config):
    # x, _ = load_data_from_train(args)
    # x = x.to(config.device)

    print(f"min in nat is: {min(eps_data)}")
    print(f"max in nat is: {max(eps_data)}")

    x_adv_eps = get_eps(x_test, model, args, config)
    adv_eps_list = []
    for adv_score in x_adv_eps:
        adv_eps_list.append(torch.norm(adv_score.flatten(), p=2).cpu().item())


    min_eps = min(min(adv_eps_list), min(eps_data))
    max_eps = max(max(adv_eps_list), max(eps_data))
    print(f"min in adv is: {min(adv_eps_list)}")
    print(f"max in adv is: {max(adv_eps_list)}")

    EPS_range = (max_eps, min_eps)
    adv_eps_list = torch.from_numpy(np.array(adv_eps_list))

    return EPS_range, adv_eps_list



# def reweight_t(x_test, model, eps_data, args, config):
#     x, _ = load_data_from_train(args)
#     x = x.to(config.device)
#     eps_data = eps_data

#     # preprocess eps_data
#     nat_eps_list = []
#     #TODO: calculation depends on the npy type loaded.
#     for score in eps_data:
#         nat_eps_list.append(torch.norm(score.flatten(), p=2).cpu().item())

#     print(f"min in nat is: {min(nat_eps_list)}")
#     print(f"max in nat is: {max(nat_eps_list)}")
#     # print(f"the length of the list is: {len(nat_eps_list)}")
#     #TODO: eps threshold depends on the method given.
#     target_range = t
#     x_adv_eps = get_eps(x_test, model, args, config)
#     adv_eps_list = []
#     for adv_score in x_adv_eps:
#         adv_eps_list.append(torch.norm(adv_score.flatten(), p=2).cpu().item())


#     min_eps = min(min(adv_eps_list), min(nat_eps_list))
#     max_eps = max(max(adv_eps_list), max(nat_eps_list))
#     print(f"min in adv is: {min(adv_eps_list)}")
#     print(f"max in adv is: {max(adv_eps_list)}")
#     EPS_range = max_eps - min_eps
#     zoom_ratio = target_range / EPS_range
#     adv_eps_list = torch.from_numpy(np.array(adv_eps_list))
#     reweighted_t = (adv_eps_list - min_eps) * zoom_ratio + (t/2 - target_range/2)
#     assert torch.all(reweighted_t > 0)
#     return reweighted_t

#     adv_eps_list - min_eps / max_eps - min_eps  * 125 + bias

#     adv_eps = min_eps + (max_eps-min_eps)/2