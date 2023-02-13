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
        args.losses_path='/scratch2/pedroroblesduten/BCOS/losses'
        args.ckpt_path='/scratch2/pedroroblesduten/BCOS/ckpt'
        args.batch_size=128
        args.device='cuda'

    elif mode == 'local':
        args.model_name='resNet34'
        args.dataset='CIFAR10'
        args.imagenetPath=r"C:\Users\pedro\OneDrive\Área de Trabalho\classical_datasets\imagenet"
        args.cifar10Path=r"C:\Users\pedro\OneDrive\Área de Trabalho\classical_datasets\CIFAR10"
        #args.cifar100Path='/scratch2/pedroroblesduten/classical_datasets/cifar100'
        args.epochs=200
        args.losses_path=r"C:\Users\pedro\OneDrive\Área de Trabalho\LOCAL\bcos\losses"
        args.ckpt_path = r"C:\Users\pedro\OneDrive\Área de Trabalho\LOCAL\bcos\ckpt"
        args.batch_size=10
        args.device='cuda'
      
    return args
