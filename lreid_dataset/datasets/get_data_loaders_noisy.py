# import torchvision.transforms as T
import copy
import os.path
import os
from reid.utils.feature_tools import *
import lreid_dataset.datasets as datasets
# from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data.sampler import RandomMultipleGallerySampler,RandomIdentitySampler
from reid.utils.data import IterLoader
import numpy as np
import os.path as osp
name_map={
    'market1501':"market", 
    'cuhk_sysu':"subcuhksysu", 
    'dukemtmc':"duke", 
    'msmt17':"msmt17", 
    'cuhk03':"cuhk03"
}
def get_data_noisy(name, data_dir, height, width, batch_size, workers, num_instances, select_num=0,noise='clean',noise_ratio=0.1):
    root = data_dir
    dataset = datasets.create(name, root)
    # create the clean data first
    if select_num > 0:
        '''select some persons for training'''
        train = []
        for instance in dataset.train:
            if instance[1] < select_num:
                train.append((instance[0], instance[1], instance[2], instance[3]))  #img_path, pid, camid, domain-id

        dataset.train = train
        dataset.num_train_pids = select_num
        dataset.num_train_imgs = len(train)
    '''using noisy labels'''
    if noise in ("random","pattern") and name in name_map.keys():
        if select_num > 0:
            file_path='noisy_data/{}_{}_{}.pt'.format(name_map[name],noise,noise_ratio)
        else:
            file_path='noisy_data_full/{}_{}_{}.pt'.format(name_map[name],noise,noise_ratio)
        print("**************\nloading noisy data form {}\n********* ".format(file_path))
        noisy_data=torch.load(file_path)
        train = []        
          
        noisy_count=0
        for idx, d in enumerate(noisy_data):    ##img_path, global_pid, global_cid, dataset_name, local_pid
            clean_pid=dataset.train[idx][1]

            img_path=osp.join(osp.dirname(dataset.train[idx][0]), d[0])

            # print(d, dataset.train[idx])
            
            train.append((img_path,d[4],d[2],dataset.train[0][3],idx, clean_pid))#img_path, pid, camid, domain-id, image-id, clean_pid
            if d[4]!=clean_pid:
                noisy_count+=1                
        print("Noisy ratio:",noisy_count/len(train))        
        dataset.train = train
        if select_num>0:
            dataset.num_train_pids = select_num
            dataset.num_train_imgs = len(train)     
    else:
        train = []                
        noisy_count=0
        for idx, d in enumerate(dataset.train):    ##img_path, global_pid, global_cid, dataset_name, local_pid            
            train.append((d[0],d[1],d[2],d[3],idx, d[1]))#img_path, pid, camid, domain-id, image-id, clean_pid             
        print("Noisy ratio:",noisy_count/len(train))        
        dataset.train = train
        if select_num>0:
            dataset.num_train_pids = select_num
            dataset.num_train_imgs = len(train)  
        
    

    # exit(0)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset.train)

    iters = int(len(train_set) / batch_size)
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])
       

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None


    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    init_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=test_transformer),
                             batch_size=128, num_workers=workers,shuffle=False, pin_memory=True, drop_last=False)

    return [dataset, num_classes, train_loader, test_loader, init_loader, name]
def get_data_purify( dataset, height, width, batch_size, workers, num_instances, Keep, Pseudo):    
    # create the clean data first
    dataset_new=copy.deepcopy(dataset)
    dataset_new.train=[]
    init_train=[]
    for flag, data, pse in zip(Keep, dataset.train, Pseudo):
        if flag:    #img_path, pid, camid, domain-id, image-id, clean_pid
            pse=int(pse)
            dataset_new.train.append((data[0], pse, data[2], data[3],data[4],data[5]))
            init_train.append(dataset_new.train[-1])
        else:
            init_train.append(data)
    print("***************maintained data ratio", len(dataset_new.train)/len(dataset.train))
    # for flag, data in zip(Keep, dataset.train):
    #     if flag:
    #         dataset_new.train.append(data)

    # print("***************maintained data ratio", len(dataset_new.train)/len(dataset.train))

   

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset_new.train)

    iters = int(len(train_set) / batch_size)
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])
       
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
        # sampler=RandomIdentitySampler(train_set, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])
    init_loader = DataLoader(Preprocessor(init_train, root=dataset.images_dir,transform=test_transformer),
                             batch_size=128, num_workers=workers,shuffle=False, pin_memory=True, drop_last=False)

   
    return train_loader, init_loader
def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def build_data_loaders_noisy(cfg, training_set, testing_only_set, select_num=500):
    # Create data loaders
    data_dir = cfg.data_dir
    height, width = (256, 128)
    training_loaders = [get_data_noisy(name, data_dir, height, width, cfg.batch_size, cfg.workers,
                                 cfg.num_instances, select_num=select_num, noise=cfg.noise, noise_ratio=cfg.noise_ratio) for name in training_set]

  
    testing_loaders = [get_data_noisy(name, data_dir, height, width, cfg.batch_size, cfg.workers,
                                cfg.num_instances) for name in testing_only_set]
    return training_loaders, testing_loaders
