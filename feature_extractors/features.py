"""
PatchCore logic based on https://github.com/rvorias/ind_knn_ad
"""

from sklearn import random_projection
from utils.utils import KNNGaussianBlur
from utils.utils import set_seeds
import numpy as np
from sklearn.metrics import roc_auc_score
import timm
import torch
from tqdm import tqdm
from utils.au_pro_util import calculate_au_pro
from feature_extractors.pointnet2_utils import *
from pointnet2_ops import pointnet2_utils
import cv2
import os
from utils.mvtec3d_util import *
import time
import open3d as o3d
# from feature_extractors.models import *
from torch.utils.data import DataLoader
from knn_cuda import KNN

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data, fps_idx


def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

class Features(torch.nn.Module):


    def unorganized_data_to_organized(self,unorganized_pc, none_zero_data_list):
        '''

        Args:
            unorganized_pc:
            none_zero_data_list:

        Returns:

        '''
        # print(none_zero_data_list[0].shape)
        if not isinstance(none_zero_data_list, list):
            none_zero_data_list = [none_zero_data_list]

        for idx in range(len(none_zero_data_list)):
            none_zero_data_list[idx] = none_zero_data_list[idx].squeeze().detach().cpu().numpy()

        # print("unorganized_pc",unorganized_pc.shape)


        unorganized_pc = unorganized_pc.numpy()
        if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
            nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
            

        full_data_list = []

        for none_zero_data in none_zero_data_list:
            if none_zero_data.ndim == 1:
                none_zero_data = np.expand_dims(none_zero_data,1)
            full_data = np.zeros((unorganized_pc.shape[0], none_zero_data.shape[1]), dtype=none_zero_data.dtype)
            
            if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
                full_data[nonzero_indices, :] = none_zero_data
            else:
                full_data = none_zero_data

            full_data_reshaped = full_data.reshape((1, unorganized_pc.shape[0], none_zero_data.shape[1]))
            full_data_tensor = torch.tensor(full_data_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
            full_data_list.append(full_data_tensor)

        return full_data_list

    def normalize(self,pred, max_value=None, min_value=None):
        if max_value is None or min_value is None:
            return (pred - pred.min()) / (pred.max() - pred.min())
        else:
            return (pred - min_value) / (max_value - min_value)


    def apply_ad_scoremap(self,image, scoremap, alpha=0.5):
        np_image = np.asarray(image, dtype=float)
        scoremap = (scoremap * 255).astype(np.uint8)
        scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
        return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)




    def __init__(self, image_size=224, f_coreset=0.1, coreset_eps=0.9,args = None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.deep_feature_extractor = Model(device=self.device)
        # self.deep_feature_extractor.to(self.device)
        # self.deep_feature_extractor.freeze_parameters(layers=[], freeze_bn=True)
        self.args = args
        self.image_size = image_size
        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.blur = KNNGaussianBlur(4)
        self.n_reweight = 3
        set_seeds(0)
        self.patch_lib = []
        self.anomaly_patch_lib = []
        self.pre_patch_lib = []
        self.tmp_patch_lib = []
        self.name_list = []
        self.test_patch_lib = []
        self.patch_lib_neighbor_scale = None


        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.gts = []
        self.predictions = []
        self.image_rocauc = 0
        self.pixel_rocauc = 0
        self.au_pro = 0

    def __call__(self, x):
        # Extract the desired feature maps using the backbone model.
        with torch.no_grad():
            feature_maps = self.deep_feature_extractor(x)

        feature_maps = [fmap.to("cpu") for fmap in feature_maps]
        return feature_maps

    def add_sample_to_mem_bank(self, sample):
        raise NotImplementedError

    def predict(self, sample, mask, label):
        raise NotImplementedError

    def init_para(self):
        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.gts = []
        self.predictions = []
        self.image_rocauc = 0
        self.pixel_rocauc = 0
        self.au_pro = 0

    def _reduce_knn_distances(self, knn_distances, knn_indices, matching_mode, temperature, consistency_weight, neighbor_scale_bank=None):
        if matching_mode == "knn_mean":
            return knn_distances.mean(dim=1)

        weights = torch.softmax(-knn_distances / temperature, dim=1)
        weighted_distance = torch.sum(weights * knn_distances, dim=1)

        if matching_mode == "distance_weighted":
            return weighted_distance

        if matching_mode == "adaptive_knn":
            if neighbor_scale_bank is not None and knn_indices is not None:
                neighbor_scale = neighbor_scale_bank[knn_indices].mean(dim=1)
            else:
                neighbor_scale = knn_distances.mean(dim=1)

            normalized_distance = weighted_distance / neighbor_scale.clamp_min(1e-6)
            consistency = knn_distances.std(dim=1, unbiased=False) / knn_distances.mean(dim=1).clamp_min(1e-6)
            return normalized_distance * (1.0 + consistency_weight * consistency)

        return knn_distances[:, 0]

    def _compute_memory_matching_scores(
        self,
        query,
        memory_bank,
        matching_mode,
        k,
        temperature,
        consistency_weight,
        neighbor_scale_bank=None,
    ):
        dist = torch.cdist(query, memory_bank)
        if matching_mode == "1nn":
            scores, _ = torch.min(dist, dim=1)
            return scores

        k = min(k, memory_bank.shape[0])
        if k <= 1:
            scores, _ = torch.min(dist, dim=1)
            return scores

        knn_distances, knn_indices = torch.topk(dist, k=k, largest=False, dim=1)
        return self._reduce_knn_distances(
            knn_distances,
            knn_indices,
            matching_mode,
            temperature,
            consistency_weight,
            neighbor_scale_bank=neighbor_scale_bank,
        )

    def _build_patch_library_statistics(self):
        matching_mode = getattr(self.args, "matching_mode", "1nn").lower()
        mmd_base_matching_mode = getattr(self.args, "mmd_base_matching_mode", "1nn").lower()
        use_adaptive_statistics = (
            matching_mode == "adaptive_knn"
            or (matching_mode == "mmd" and mmd_base_matching_mode == "adaptive_knn")
        )

        if not use_adaptive_statistics:
            self.patch_lib_neighbor_scale = None
            return

        density_k = min(getattr(self.args, "matching_density_k", 5), max(self.patch_lib.shape[0] - 1, 1))
        if density_k <= 0 or self.patch_lib.shape[0] < 2:
            self.patch_lib_neighbor_scale = None
            return

        with torch.no_grad():
            lib_dist = torch.cdist(self.patch_lib, self.patch_lib)
            lib_dist.fill_diagonal_(float("inf"))
            neighbor_distances, _ = torch.topk(lib_dist, k=density_k, largest=False, dim=1)
            self.patch_lib_neighbor_scale = neighbor_distances.mean(dim=1).clamp_min(1e-6)

    def _compute_patch_matching_scores(self, patch, matching_mode=None):
        matching_mode = getattr(self.args, "matching_mode", "1nn").lower() if matching_mode is None else matching_mode
        return self._compute_memory_matching_scores(
            patch,
            self.patch_lib,
            matching_mode,
            getattr(self.args, "matching_k", 5),
            max(getattr(self.args, "matching_temperature", 1.0), 1e-6),
            max(getattr(self.args, "matching_consistency_weight", 0.5), 0.0),
            neighbor_scale_bank=self.patch_lib_neighbor_scale,
        )

    def _compute_image_level_score(self, s_map):
        s = torch.max(s_map)

        if self.args.dataset == 'real':
            s = torch.mean(s_map)
        if self.args.dataset == 'shapenet':
            tmp_s, _ = torch.topk(s_map, 80)
            s = torch.mean(tmp_s)
        if self.args.dataset == 'mulsen' or self.args.dataset == 'minishift' or self.args.dataset == 'quan':
            tmp_s, _ = torch.topk(s_map, 80)
            s = torch.mean(tmp_s)

        return s

    def _estimate_mmd_sigma(self, x_set, y_set):
        sigma = float(getattr(self.args, "mmd_sigma", 1.0))
        if sigma > 0:
            return sigma

        sample_x = x_set[: min(64, x_set.shape[0])]
        sample_y = y_set[: min(64, y_set.shape[0])]
        joined = torch.cat([sample_x, sample_y], dim=0)
        if joined.shape[0] < 2:
            return 1.0

        with torch.no_grad():
            pairwise = torch.cdist(joined, joined)
            pairwise = pairwise[pairwise > 0]
            if pairwise.numel() == 0:
                return 1.0
            return pairwise.median().item()

    def _normalize_ard_weights(self, relevance):
        eps = max(float(getattr(self.args, "ard_eps", 1e-6)), 1e-12)
        min_weight = max(float(getattr(self.args, "ard_min_weight", 0.0)), 0.0)
        weight_norm = getattr(self.args, "ard_weight_norm", "softmax").lower()

        if weight_norm == "l1":
            weights = relevance + eps
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)
        else:
            temperature = max(float(getattr(self.args, "ard_temperature", 1.0)), eps)
            weights = torch.softmax(relevance / temperature, dim=-1)

        if min_weight > 0:
            weights = torch.clamp(weights, min=min_weight)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)

        return weights

    def _compute_ard_weights(self, x_set, y_set):
        eps = max(float(getattr(self.args, "ard_eps", 1e-6)), 1e-12)

        if x_set.dim() == 2:
            x_set = x_set.unsqueeze(0)
        if y_set.dim() == 2:
            y_set = y_set.unsqueeze(0)

        x_mean = x_set.mean(dim=-2)
        y_mean = y_set.mean(dim=-2)
        y_std = y_set.std(dim=-2, unbiased=False)

        relevance = torch.abs(x_mean - y_mean) / (y_std + eps)
        return self._normalize_ard_weights(relevance)

    def _compute_ard_rbf_kernel(self, x_set, y_set, ard_weights, sigma):
        if x_set.dim() == 2:
            x_set = x_set.unsqueeze(0)
        if y_set.dim() == 2:
            y_set = y_set.unsqueeze(0)
        if ard_weights.dim() == 1:
            ard_weights = ard_weights.unsqueeze(0)

        diff_sq = (x_set.unsqueeze(-2) - y_set.unsqueeze(-3)) ** 2
        weighted_dist_sq = (diff_sq * ard_weights.unsqueeze(-2).unsqueeze(-2)).sum(dim=-1)
        kernel = torch.exp(-weighted_dist_sq / (2 * (max(sigma, 1e-6) ** 2)))
        return kernel.squeeze(0) if kernel.shape[0] == 1 else kernel

    def _compute_kernel(self, x_set, y_set, sigma, ard_weights=None):
        kernel_type = getattr(self.args, "mmd_kernel", "rbf").lower()

        if kernel_type == "ard_rbf":
            if ard_weights is None:
                ard_weights = self._compute_ard_weights(x_set, y_set)
            return self._compute_ard_rbf_kernel(x_set, y_set, ard_weights, sigma)

        if kernel_type == "cosine":
            x_norm = torch.nn.functional.normalize(x_set, dim=-1)
            y_norm = torch.nn.functional.normalize(y_set, dim=-1)
            return torch.matmul(x_norm, y_norm.transpose(-1, -2)).clamp(-1.0, 1.0)

        if kernel_type == "laplacian":
            dist = torch.cdist(x_set, y_set)
            return torch.exp(-dist / max(sigma, 1e-6))

        if kernel_type == "polynomial":
            degree = max(int(getattr(self.args, "mmd_poly_degree", 2)), 1)
            coef0 = float(getattr(self.args, "mmd_poly_coef0", 1.0))
            return (torch.matmul(x_set, y_set.transpose(-1, -2)) + coef0) ** degree

        if kernel_type in ("rq", "rational_quadratic", "rational-quadratic"):
            alpha = max(float(getattr(self.args, "mmd_rq_alpha", 1.0)), 1e-6)
            dist_sq = torch.cdist(x_set, y_set) ** 2
            return (1.0 + dist_sq / (2.0 * alpha * max(sigma, 1e-6) ** 2)) ** (-alpha)

        dist_sq = torch.cdist(x_set, y_set) ** 2
        return torch.exp(-dist_sq / (2 * (max(sigma, 1e-6) ** 2)))

    def _compute_mmd_score(self, x_set, y_set, sigma, ard_weights=None):
        k_xx = self._compute_kernel(x_set, x_set, sigma, ard_weights=ard_weights)
        k_yy = self._compute_kernel(y_set, y_set, sigma, ard_weights=ard_weights)
        k_xy = self._compute_kernel(x_set, y_set, sigma, ard_weights=ard_weights)

        mmd_sq = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
        return torch.sqrt(torch.clamp(mmd_sq, min=0.0) + 1e-12)

    def _select_mmd_reference_subset(self, patch, local_patch_scores):
        reference_mode = getattr(self.args, "mmd_reference_mode", "topk_nn").lower()
        max_reference = min(getattr(self.args, "mmd_k", 128), self.patch_lib.shape[0])
        if max_reference <= 0:
            return self.patch_lib[:1]

        if reference_mode != "topk_nn":
            return self.patch_lib[:max_reference]

        query_count = min(
            patch.shape[0],
            max(4, min(32, max_reference // 4 if max_reference >= 4 else max_reference)),
        )
        if query_count <= 0:
            return self.patch_lib[:max_reference]

        _, query_indices = torch.topk(local_patch_scores, k=query_count, largest=True)
        focused_patch = patch[query_indices]

        candidate_k = min(self.patch_lib.shape[0], max(1, int(np.ceil(max_reference / query_count))))
        candidate_distances = torch.cdist(focused_patch, self.patch_lib)
        knn_distances, knn_indices = torch.topk(candidate_distances, k=candidate_k, largest=False, dim=1)

        flat_indices = knn_indices.reshape(-1).detach().cpu().tolist()
        flat_distances = knn_distances.reshape(-1).detach().cpu().tolist()
        sorted_pairs = sorted(zip(flat_distances, flat_indices), key=lambda item: item[0])

        selected = []
        visited = set()
        for _, idx in sorted_pairs:
            if idx in visited:
                continue
            visited.add(idx)
            selected.append(idx)
            if len(selected) >= max_reference:
                break

        if not selected:
            return self.patch_lib[:max_reference]

        selected_idx = torch.tensor(selected, device=self.patch_lib.device, dtype=torch.long)
        return self.patch_lib[selected_idx]

    def _compute_mmd_matching_score(self, patch, local_patch_scores):
        reference_set = self._select_mmd_reference_subset(patch, local_patch_scores)
        sigma = self._estimate_mmd_sigma(patch, reference_set)
        ard_weights = None
        if getattr(self.args, "mmd_kernel", "rbf").lower() == "ard_rbf":
            ard_weights = self._compute_ard_weights(patch, reference_set)
        return self._compute_mmd_score(patch, reference_set, sigma, ard_weights=ard_weights)

    def _compute_local_patch_neighbors(self, patch, center):
        local_k = min(max(1, getattr(self.args, "mmd_local_k", 7)), patch.shape[0])
        if center is not None:
            center = center.squeeze(0) if center.dim() == 3 else center
            patch_distance = torch.cdist(center, center)
        else:
            patch_distance = torch.cdist(patch, patch)

        _, local_indices = torch.topk(patch_distance, k=local_k, largest=False, dim=1)
        return local_indices

    def _normalize_mmd_scores(self, scores):
        norm_mode = getattr(self.args, "mmd_norm", "zscore").lower()
        if norm_mode == "none":
            return scores

        if norm_mode == "minmax":
            score_min = scores.min()
            score_max = scores.max()
            denom = (score_max - score_min).clamp_min(1e-6)
            return (scores - score_min) / denom

        mean = scores.mean()
        std = scores.std(unbiased=False).clamp_min(1e-6)
        normalized = (scores - mean) / std
        return torch.relu(normalized)

    def _compute_local_mmd_scores(self, patch, center):
        local_indices = self._compute_local_patch_neighbors(patch, center)
        local_patch_sets = patch[local_indices]

        ref_k = min(max(1, getattr(self.args, "mmd_ref_k", 8)), self.patch_lib.shape[0])
        ref_distances = torch.cdist(patch, self.patch_lib)
        _, ref_indices = torch.topk(ref_distances, k=ref_k, largest=False, dim=1)
        reference_sets = self.patch_lib[ref_indices]

        sigma = self._estimate_mmd_sigma(
            local_patch_sets.reshape(-1, local_patch_sets.shape[-1]),
            reference_sets.reshape(-1, reference_sets.shape[-1]),
        )

        ard_weights = None
        if getattr(self.args, "mmd_kernel", "rbf").lower() == "ard_rbf":
            ard_weights = self._compute_ard_weights(local_patch_sets, reference_sets)

        k_xx = self._compute_kernel(local_patch_sets, local_patch_sets, sigma, ard_weights=ard_weights)
        k_yy = self._compute_kernel(reference_sets, reference_sets, sigma, ard_weights=ard_weights)
        k_xy = self._compute_kernel(local_patch_sets, reference_sets, sigma, ard_weights=ard_weights)

        mmd_sq = k_xx.mean(dim=(1, 2)) + k_yy.mean(dim=(1, 2)) - 2.0 * k_xy.mean(dim=(1, 2))
        return torch.sqrt(torch.clamp(mmd_sq, min=0.0) + 1e-12)


    def compute_anomay_scores(self, patch, mask, label, path, unorganized_pc, unorganized_pc_no_zeros, center):

        feature_map_dims = patch.shape[0]
        matching_mode = getattr(self.args, "matching_mode", "1nn").lower()
        mmd_mode = getattr(self.args, "mmd_mode", "global").lower()
        local_matching_mode = (
            getattr(self.args, "mmd_base_matching_mode", "adaptive_knn").lower()
            if matching_mode == "mmd"
            else matching_mode
        )
        patch_scores = self._compute_patch_matching_scores(patch, matching_mode=local_matching_mode)

        if matching_mode == "mmd" and mmd_mode == "local":
            local_mmd_scores = self._compute_local_mmd_scores(patch, center)
            local_mmd_scores = self._normalize_mmd_scores(local_mmd_scores)
            patch_blend = max(getattr(self.args, "mmd_patch_blend", 0.3), 0.0)
            patch_scores = patch_scores + patch_blend * local_mmd_scores

        s_map = patch_scores.view(1, 1, feature_map_dims)


        if self.args.use_LFSA:
            s_map = interpolating_points_chunked(unorganized_pc_no_zeros.permute(0,2,1).to(self.args.device), center.permute(0,2,1).to(self.args.device), s_map.to(self.args.device)).permute(0,2,1)
            s_map = torch.Tensor(self.unorganized_data_to_organized(unorganized_pc, [s_map])[0]).to(self.args.device)

            if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
                s_map = s_map.squeeze().reshape(1,224,224)
                s_map = self.blur(s_map)

            else:
                num_group = 1024
                group_size = 12

                batch_size, num_points, _ = unorganized_pc_no_zeros.contiguous().shape
                center, center_idx = fps(unorganized_pc_no_zeros.contiguous(), num_group)  # B G 3

                # knn to get the neighborhood
                knn = KNN(k=group_size, transpose_mode=True)
                _, idx = knn(unorganized_pc_no_zeros, center)  # B G M

                ori_idx = idx
                idx_base = torch.arange(0, batch_size, device=unorganized_pc_no_zeros.device).view(-1, 1, 1) * num_points
                
                idx = idx + idx_base
                idx = idx.view(-1)
                neighborhood = s_map.reshape(batch_size * num_points, -1)[idx, :]
                neighborhood = neighborhood.reshape(batch_size, num_group, group_size, -1).contiguous()
                agg_s_map = torch.mean(neighborhood,-2).view(1, 1, -1)

                s_map = interpolating_points_chunked(unorganized_pc_no_zeros.permute(0,2,1).to(self.args.device), center.permute(0,2,1).to(self.args.device), agg_s_map.to(self.args.device)).permute(0,2,1)
                s_map = torch.Tensor(self.unorganized_data_to_organized(unorganized_pc, [s_map])[0]).to(self.args.device)

        s_map = s_map.squeeze(0)
        s = self._compute_image_level_score(s_map)

        if matching_mode == "mmd" and mmd_mode == "global":
            mmd_score = self._compute_mmd_matching_score(patch, patch_scores)
            blend = float(np.clip(getattr(self.args, "mmd_blend", 0.3), 0.0, 1.0))
            s = (1.0 - blend) * s + blend * mmd_score
        

        if self.args.vis_save:
            while isinstance(path,list):
                path = path[0]
            from pathlib import Path
            parts = path.split("data", 1) 
            post_data_path = parts[1].lstrip(os.sep) 
            save_path = "./vis-results/"+post_data_path

            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            scoremap = normalize(s_map.squeeze())

            scoremap = (scoremap.cpu().numpy() * 255).astype(np.uint8)
            scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
            scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
            unorganized_pc = unorganized_pc.squeeze().cpu()
            scoremap = torch.Tensor(scoremap).squeeze()
            outpoints = torch.cat([unorganized_pc,scoremap],1)

            save_path = str(Path(save_path).with_suffix(".txt"))
            np.savetxt(save_path, outpoints.numpy())
            save_path = "./vis-results-GT/"+post_data_path

            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            scoremap = scoremap.cpu().numpy().astype(np.uint8)
            scoremap[mask.flatten().numpy()==1]=np.array([255,0,0])
            scoremap[mask.flatten().numpy()==0]=np.array([0,0,255])
            scoremap = torch.Tensor(scoremap).squeeze()
            outpoints = torch.cat([unorganized_pc,scoremap],1)
            save_path = str(Path(save_path).with_suffix(".txt"))
            np.savetxt(save_path, outpoints.numpy())


        self.image_preds.append(s.cpu().numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.cpu().flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())

        self.predictions.append(s_map.squeeze().detach().cpu().squeeze().numpy())
        self.gts.append(mask.squeeze().detach().cpu().squeeze().numpy())








    def calculate_metrics(self,path=None):
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.array(self.pixel_preds)

        if not path == None:
            numpy_save = normalize(self.image_preds)
            numpy_save = (numpy_save * 255).astype(np.uint8)
            numpy_save_gt = (self.image_labels*255).astype(np.uint8)[:,0]
            numpy_save = np.append(numpy_save, numpy_save_gt, axis=0)
            np.save(path, numpy_save)


        self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
        if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
            self.au_pro, _ = calculate_au_pro(self.gts, self.predictions)
        else:
            self.au_pro = 0



    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0).cpu()
        n = len(self.patch_lib)

        self.f_coreset = 0.05
        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            
            self.patch_lib = self.patch_lib[self.coreset_idx].to(self.args.device)
        else:
            self.patch_lib = self.patch_lib.to(self.args.device)

        self._build_patch_library_statistics()

               

    def get_coreset_idx_randomp(self, z_lib, n=1000, eps=0.90, float16=True, force_cpu=False):
        """Returns n coreset idx for given z_lib.
        Performance on AMD3700, 32GB RAM, RTX3080 (10GB):
        CPU: 40-60 it/s, GPU: 500+ it/s (float32), 1500+ it/s (float16)
        Args:
            z_lib:      (n, d) tensor of patches.
            n:          Number of patches to select.
            eps:        Agression of the sparse random projection.
            float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
            force_cpu:  Force cpu, useful in case of GPU OOM.
        Returns:
            coreset indices
        """

        print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
        try:
            target_dim = random_projection.johnson_lindenstrauss_min_dim(
                n_samples=z_lib.shape[0],
                eps=eps,
            )
            if target_dim < z_lib.shape[1]:
                transformer = random_projection.SparseRandomProjection(n_components=target_dim, eps=eps)
                z_lib = torch.tensor(transformer.fit_transform(z_lib))
                print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
            else:
                print(
                    f"   Skipping random projection because target dim {target_dim} "
                    f"is not smaller than feature dim {z_lib.shape[1]}."
                )
        except ValueError:
            print("   Error: could not project vectors. Please increase `eps`.")

        select_idx = 0
        last_item = z_lib[select_idx:select_idx + 1]
        coreset_idx = [torch.tensor(select_idx)]
        min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
        # The line below is not faster than linalg.norm, although i'm keeping it in for
        # future reference.
        # min_distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True)

        if float16:
            last_item = last_item.half()
            z_lib = z_lib.half()
            min_distances = min_distances.half()
        if torch.cuda.is_available() and not force_cpu:
            last_item = last_item.to("cuda")
            z_lib = z_lib.to("cuda")
            min_distances = min_distances.to("cuda")

        for _ in tqdm(range(n - 1)):
            distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)  # broadcasting step
            min_distances = torch.minimum(distances, min_distances)  # iterative step
            select_idx = torch.argmax(min_distances)  # selection step

            # bookkeeping
            last_item = z_lib[select_idx:select_idx + 1]
            min_distances[select_idx] = 0
            coreset_idx.append(select_idx.to("cpu"))
        return torch.stack(coreset_idx)
