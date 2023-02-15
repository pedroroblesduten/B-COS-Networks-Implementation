import argparse

def getArgs(mode):
    parser = argparse.ArgumentParser(description="BCOS_TRAINING")
    args = parser.parse_args()
    if mode == 'speed':
        args.model_name='resNet34'
        args.dataset='CIFAR10'
        args.imagenetPath='/scratch2/pedroroblesduten/classical_datasets/imagenet'
        args.cifar10Path='/scratch2/pedroroblesduten/classical_datasets/cifar10'
        args.cifar100Path='/scratch2/pedroroblesduten/classical_datasets/cifar100'
        args.epochs=200
        args.save_losses='/scratch2/pedroroblesduten/BCOS/losses'
        args.save_ckpt='/scratch2/pedroroblesduten/BCOS/ckpt'
        args.load_ckpt='/scratch2/pedroroblesduten/BCOS/ckpt/BCOS_bestVAL.pt'
        args.save_results_path = '/scratch2/pedroroblesduten/BCOS/B-COS-Networks-Implementation/results'
        args.batch_size=128
        args.device='cuda'
        args.verbose = False

    elif mode == 'local':
        args.model_name='resNet34'
        args.dataset='CIFAR10'
        args.imagenetPath="C:/Users/pedro/OneDrive/Área de Trabalho/classical_datasets/imagenet"
        args.cifar10Path="C:/Users/pedro/OneDrive/Área de Trabalho/classical_datasets/CIFAR10"
        #args.cifar100Path='/scratch2/pedroroblesduten/classical_datasets/cifar100'
        args.epochs=200
        args.save_losses="./losses"
        args.save_ckpt = "/ckpt"
        args.batch_size=32
        args.device='cuda'
        args.verbose = True
      
    return args
