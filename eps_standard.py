import torch

def load_dataset_eps(args):
    if args.dataset == 'cifar10':
        if args.use_score:
            print("Using Score Norm as reweighting metric!")
            cf10_pt = f"Enter the path to the score file by running cifar10_eps_range.sh"
        else:
            cf10_pt = f"Enter the path to the eps file by running cifar10_eps_range.sh"
        cf10_eps_data = torch.load(cf10_pt)

        cf10_nat_eps_list = []
        for score in cf10_eps_data:
            cf10_nat_eps_list.append(torch.norm(score.flatten(), p=2).cpu().item())

        cf10_eps_standard = cf10_nat_eps_list
        return cf10_eps_standard
    elif args.dataset == 'imagenet':
        imgnet_pt = f"Enter the path to the eps file by running imagenet_eps_range.sh"
        imgnet_eps_data = torch.load(imgnet_pt)

        imgnet_nat_eps_list = []
        for score in imgnet_eps_data:
            imgnet_nat_eps_list.append(torch.norm(score.flatten(), p=2).cpu().item())

        imagenet_eps_standard = imgnet_nat_eps_list
        return imagenet_eps_standard
    else:
        raise NotImplementedError("Unknown dataset.")