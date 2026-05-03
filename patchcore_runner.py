from data.mvtec3d import get_mvtec_loader,mvtec3d_classes
from data.real3d import get_real_loader,real3d_classes
from data.anomalyshape import get_shapenet_loader,shapenet3d_classes
from data.MulSen import get_mulsen_loader,mulsen_classes
from data.MiniShiftAD import get_minishift_loader,minishiftAD_classes
import torch
from tqdm import tqdm
from feature_extractors.FPFH import FPFHFeatures
import numpy as np
import os
from feature_extractors.pointnet2_utils import *


class PatchCore():
    def __init__(self, ckp = '', image_size=224, args=None):
        self.args = args
        self.method  = FPFHFeatures(args=args)
        self.dataset_name = self.args.dataset
        self.level = self.args.level

    def get_dataloader(self,dataset_name,split,class_name,level='ALL'):
        if dataset_name == 'mvtec':
            return get_mvtec_loader(split, class_name=class_name)
        if dataset_name == 'real':
            return get_real_loader(split, class_name=class_name)
        if dataset_name == 'shapenet':
            return get_shapenet_loader(split, class_name=class_name)
        if dataset_name == 'mulsen':
            return get_mulsen_loader(split, class_name=class_name)
        if dataset_name == 'minishift':
            return get_minishift_loader(split, class_name=class_name,level=level)  

    def fit(self, class_name):
        train_loader = self.get_dataloader(self.dataset_name,'train',class_name,level=self.level)
        for pc, _, _, path in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):
            self.method.collect_features(pc)
            self.method.name_list.append(path)
        print(f'\n\nRunning coreset on class {class_name}...')
        self.method.run_coreset()

    def train_point_offset(self, class_name):
        if not getattr(self.args, "use_point_offset_head", False) or not getattr(self.args, "point_offset_train", False):
            return

        train_split = getattr(self.args, "point_offset_train_split", "test")
        epochs = max(int(getattr(self.args, "point_offset_epochs", 1)), 1)
        use_cache = getattr(self.args, "point_offset_cache_train", True)

        cached_samples = None
        if use_cache:
            cached_samples = []
            cache_loader = self.get_dataloader(self.dataset_name, train_split, class_name, level=self.level)
            for pc, mask, _, path in tqdm(cache_loader, desc=f'Caching point-offset inputs for class {class_name}'):
                if not torch.is_tensor(mask) or mask.ndim < 2:
                    continue
                sample = self.method.build_point_offset_training_sample(pc, mask, path)
                if sample is not None:
                    cached_samples.append(sample)

            print(f'Cached {len(cached_samples)} point-offset samples for class {class_name}.')

        for epoch in range(epochs):
            losses = []
            if cached_samples is not None:
                iterator = tqdm(
                    cached_samples,
                    desc=f'Training point-offset head for class {class_name} [epoch {epoch + 1}/{epochs}]',
                )
                for input_features, base_scores, target in iterator:
                    loss = self.method.train_point_offset_from_tensors(input_features, base_scores, target)
                    if loss is not None:
                        losses.append(loss)
            else:
                train_loader = self.get_dataloader(self.dataset_name, train_split, class_name, level=self.level)
                for pc, mask, _, path in tqdm(
                    train_loader,
                    desc=f'Training point-offset head for class {class_name} [epoch {epoch + 1}/{epochs}]',
                ):
                    if not torch.is_tensor(mask) or mask.ndim < 2:
                        continue
                    loss = self.method.train_point_offset(pc, mask, path)
                    if loss is not None:
                        losses.append(loss)

            if losses:
                print(f'Point-offset head epoch {epoch + 1}/{epochs}, class {class_name}, loss: {float(np.mean(losses)):.4f}')


    def evaluate(self, class_name):
        image_rocaucs = dict()
        pixel_rocaucs = dict()
        au_pros = dict()
        test_loader = self.get_dataloader(self.dataset_name,'test',class_name,level=self.level)
        with torch.no_grad():
            self.method.init_para()
            self.method.name_list = []
            self.method.test_patch_lib = []
        
            for pc, mask, label, path in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
                self.method.predict(pc, mask, label,path)
        method_name = "Simple3D"
        self.method.calculate_metrics()
        image_rocaucs[method_name] = round(self.method.image_rocauc, 3)
        pixel_rocaucs[method_name] = round(self.method.pixel_rocauc, 3)
        au_pros[method_name] = round(self.method.au_pro, 3)
        print(
            f'Class: {class_name}, {method_name} Image ROCAUC: {self.method.image_rocauc:.3f}, {method_name} Pixel ROCAUC: {self.method.pixel_rocauc:.3f}, {method_name} AU-PRO: {self.method.au_pro:.3f}')
        return image_rocaucs, pixel_rocaucs, au_pros
