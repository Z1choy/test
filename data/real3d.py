import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
DATASETS_PATH = os.environ.get('REAL3D_DATASETS_PATH', '/media/chenyu/PASDF_/Real3D-AD-PCD')
POINT_FILE_PATTERNS = ("*.pcd", "*.asc", "*.txt")
GT_FILE_PATTERNS = ("*.pcd", "*.asc", "*.txt", "*.npy")


def _collect_point_files(candidate_dirs):
    point_paths = []
    seen_paths = set()
    for candidate_dir in candidate_dirs:
        if not os.path.isdir(candidate_dir):
            continue
        for pattern in POINT_FILE_PATTERNS:
            for point_path in glob.glob(os.path.join(candidate_dir, pattern)):
                if point_path not in seen_paths:
                    seen_paths.add(point_path)
                    point_paths.append(point_path)
    point_paths.sort()
    return point_paths


def _collect_files(candidate_dirs, patterns):
    file_paths = []
    seen_paths = set()
    for candidate_dir in candidate_dirs:
        if not os.path.isdir(candidate_dir):
            continue
        for pattern in patterns:
            for file_path in glob.glob(os.path.join(candidate_dir, pattern)):
                if file_path not in seen_paths:
                    seen_paths.add(file_path)
                    file_paths.append(file_path)
    file_paths.sort()
    return file_paths


def _read_point_file(point_path):
    ext = os.path.splitext(point_path)[1].lower()
    if ext == ".npy":
        points = np.load(point_path).astype(np.float32)
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        return points
    if ext == ".pcd":
        pcd = o3d.io.read_point_cloud(point_path)
        return np.asarray(pcd.points, dtype=np.float32)

    points = np.loadtxt(point_path, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    return points


def _find_matching_gt_path(test_path, test_root, gt_root):
    if not os.path.isdir(gt_root):
        return 0

    stem = os.path.splitext(os.path.basename(test_path))[0]
    relative_path = os.path.relpath(test_path, test_root)
    relative_stem = os.path.splitext(relative_path)[0]

    candidate_dirs = [
        os.path.dirname(os.path.join(gt_root, relative_path)),
        os.path.join(gt_root, os.path.dirname(relative_stem)),
        gt_root,
    ]
    for candidate_dir in candidate_dirs:
        for pattern in GT_FILE_PATTERNS:
            ext = pattern.replace("*", "")
            candidate_path = os.path.join(candidate_dir, stem + ext)
            if os.path.exists(candidate_path):
                return candidate_path

    recursive_matches = []
    for pattern in GT_FILE_PATTERNS:
        ext = pattern.replace("*", "")
        recursive_matches.extend(glob.glob(os.path.join(gt_root, "**", stem + ext), recursive=True))
    recursive_matches.sort()
    return recursive_matches[0] if recursive_matches else 0


def _downsample_points_with_mask(input_points, point_mask=None, gt_points=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(input_points[:, 0:3])
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size_setting)
    down_points = np.asarray(pcd_down.points)

    if down_points.shape[0] == 0:
        return down_points, torch.zeros([1, 0])

    mask = np.zeros(down_points.shape[0], dtype=np.float32)
    if point_mask is not None:
        point_mask = np.asarray(point_mask).reshape(-1)
        if point_mask.shape[0] == input_points.shape[0]:
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            for i, point in enumerate(down_points):
                [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
                if k > 0:
                    mask[i] = point_mask[idx[0]]
    elif gt_points is not None and len(gt_points) > 0:
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_points[:, 0:3])
        gt_tree = o3d.geometry.KDTreeFlann(gt_pcd)
        match_radius_sq = (voxel_size_setting * 1.5) ** 2
        for i, point in enumerate(down_points):
            [k, _, dist] = gt_tree.search_knn_vector_3d(point, 1)
            if k > 0 and dist[0] <= match_radius_sq:
                mask[i] = 1.0

    mask = torch.tensor(mask)
    mask = torch.where(mask > 0.5, 1., .0).unsqueeze(0)
    return down_points, mask


def _load_gt_for_points(gt_path, input_points):
    if gt_path == 0:
        return None, None

    gt_data = _read_point_file(gt_path)
    if gt_data.size == 0:
        return None, None

    gt_data = np.asarray(gt_data, dtype=np.float32)
    if gt_data.ndim == 1:
        gt_data = gt_data.reshape(-1, 1)

    if gt_data.shape[0] == input_points.shape[0] and gt_data.shape[1] == 1:
        return gt_data.reshape(-1), None
    if gt_data.shape[0] == input_points.shape[0] and gt_data.shape[1] >= 4:
        return gt_data[:, -1], None
    if gt_data.shape[1] >= 3:
        return None, gt_data[:, 0:3]

    return None, None

def real3d_classes():
    return [
        "airplane",   
        "candybar",    
        "car",         
        "chicken",     
        "diamond",      
        "duck",         
        "fish",        
        "gemstone",
        "seahorse",
        "shell",
        "starfish",
        "toffees",
    ]

voxel_size_setting = 0.15
class Real3D(Dataset):

    def __init__(self, split, class_name):
        self.cls = class_name
        self.data_path = os.path.join(DATASETS_PATH, self.cls, split)


class Real3DTrain(Real3D):
    def __init__(self, class_name):
        super().__init__(split="train", class_name=class_name)
        self.pcd_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        pcd_tot_paths = []
        tot_labels = []

        name = self.cls
        candidate_dirs = [
            os.path.join(DATASETS_PATH, name, 'train'),
            os.path.join(DATASETS_PATH, name, 'train_cut'),
            os.path.join(DATASETS_PATH, name, 'train', 'good'),
        ]
        pcd_paths = _collect_point_files(candidate_dirs)
        pcd_tot_paths.extend(pcd_paths)
        tot_labels.extend([0] * len(pcd_paths))
        if len(pcd_paths) == 0:
            searched_dirs = ', '.join(candidate_dirs)
            print(f"[Real3D] No training point files found for class {name}. "
                  f"DATASETS_PATH={DATASETS_PATH}. Searched: {searched_dirs}. "
                  f"Patterns: {POINT_FILE_PATTERNS}")
        return pcd_tot_paths, tot_labels

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, label = self.pcd_paths[idx], self.labels[idx]
        input_points = _read_point_file(pcd_path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(input_points[:,0:3]) #点云数据
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size_setting) 

        resized_organized_pc = np.asarray(pcd.points)
        unorganized_pc = resized_organized_pc


        return unorganized_pc, label, label, pcd_path


class Real3DTest(Real3D):
    def __init__(self, class_name):
        super().__init__(split="test", class_name=class_name)
        self.pcd_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        pcd_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        gt_root = os.path.join(DATASETS_PATH, self.cls, 'gt')
        direct_pcd_paths = _collect_point_files([self.data_path])

        if direct_pcd_paths:
            for pcd_path in direct_pcd_paths:
                gt_path = _find_matching_gt_path(pcd_path, self.data_path, gt_root)
                pcd_tot_paths.append(pcd_path)
                gt_tot_paths.append(gt_path)
                tot_labels.append(0 if gt_path == 0 else 1)
        else:
            defect_types = os.listdir(self.data_path)
            for defect_type in defect_types:
                defect_dir = os.path.join(self.data_path, defect_type)
                if not os.path.isdir(defect_dir):
                    continue
                pcd_paths = _collect_point_files([defect_dir])
                if defect_type == 'good':
                    pcd_tot_paths.extend(pcd_paths)
                    gt_tot_paths.extend([0] * len(pcd_paths))
                    tot_labels.extend([0] * len(pcd_paths))
                else:
                    gt_paths = [_find_matching_gt_path(pcd_path, self.data_path, gt_root) for pcd_path in pcd_paths]
                    pcd_tot_paths.extend(pcd_paths)
                    gt_tot_paths.extend(gt_paths)
                    tot_labels.extend([1] * len(pcd_paths))

        assert len(pcd_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return pcd_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, gt, label = self.pcd_paths[idx], self.gt_paths[idx], self.labels[idx]

        input_points = _read_point_file(pcd_path)
        if gt != 0:
            gt_mask, gt_points = _load_gt_for_points(gt, input_points)
        else:
            gt_mask, gt_points = None, None

        if input_points.shape[1] < 4:
            unorganized_pc, gt_tensor = _downsample_points_with_mask(
                input_points,
                point_mask=gt_mask,
                gt_points=gt_points,
            )
            return unorganized_pc, gt_tensor[:1], label, pcd_path

        if gt_mask is not None or gt_points is not None:
            unorganized_pc, gt_tensor = _downsample_points_with_mask(
                input_points,
                point_mask=gt_mask,
                gt_points=gt_points,
            )
            return unorganized_pc, gt_tensor[:1], label, pcd_path

        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        idx1 = input_points[:,3]==0
        idx2 = input_points[:,3]==1
        pcd1.points = o3d.utility.Vector3dVector(input_points[idx1,0:3]) 
        pcd2.points = o3d.utility.Vector3dVector(input_points[idx2,0:3]) 

        pcd = pcd1 + pcd2
        pcd_new = pcd.voxel_down_sample(voxel_size=voxel_size_setting)
        
        pcd1_new = o3d.geometry.PointCloud()
        pcd2_new = o3d.geometry.PointCloud()

        pcd1_vec = []
        pcd2_vec = []
        from scipy.spatial.distance import cdist

        pcd_points_new = np.asarray(pcd_new.points)
        pcd_points = np.asarray(pcd.points)


        pc_len = len(pcd1.points)
        
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        for x in pcd_points_new:
            [k, idx_, _] = pcd_tree.search_knn_vector_3d(x, 1)
            # print(idx_[0])
            if idx_[0] < pc_len:
                pcd1_vec.append(x)
            else:
                pcd2_vec.append(x)

        pcd1.points = o3d.utility.Vector3dVector(np.asarray(pcd1_vec))
        if not len(pcd2_vec)==0:
            pcd2.points = o3d.utility.Vector3dVector(np.asarray(pcd2_vec))

       

        pcd = pcd1 + pcd2

        resized_organized_pc = np.asarray(pcd.points)
        unorganized_pc = resized_organized_pc


        if gt == 0:
            gt = torch.zeros(
                [1, unorganized_pc.shape[0]])
        else:
            gt = torch.zeros(unorganized_pc.shape[0])
            gt[len(pcd1.points):len(pcd.points)] = 1.0
            gt = torch.where(gt > 0.5, 1., .0)
            gt = gt.unsqueeze(0)
            gt = gt.unsqueeze(0)
        return unorganized_pc, gt[:1], label, pcd_path


def get_real_loader(split, class_name):
    if split in ['train']:
        dataset = Real3DTrain(class_name=class_name)
    elif split in ['test']:
        dataset = Real3DTest(class_name=class_name)

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                             pin_memory=True)
    return data_loader

