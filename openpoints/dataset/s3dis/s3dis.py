import math
import os
import pickle
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from pykdtree.kdtree import KDTree
from ..data_util import crop_pc, voxelize
from ..build import DATASETS
from ...utils import GaussianOptions, NaiveGaussian3D, fps_sample, make_gs_points, merge_gs_list


@DATASETS.register_module()
class S3DIS(Dataset):
    classes = ['ceiling',
               'floor',
               'wall',
               'beam',
               'column',
               'window',
               'door',
               'chair',
               'table',
               'bookcase',
               'sofa',
               'board',
               'clutter']
    num_classes = 13
    num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                              650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
    class2color = {'ceiling':     [0, 255, 0],
                   'floor':       [0, 0, 255],
                   'wall':        [0, 255, 255],
                   'beam':        [255, 255, 0],
                   'column':      [255, 0, 255],
                   'window':      [100, 100, 255],
                   'door':        [200, 200, 100],
                   'table':       [170, 120, 200],
                   'chair':       [255, 0, 0],
                   'sofa':        [200, 100, 100],
                   'bookcase':    [10, 200, 100],
                   'board':       [200, 200, 200],
                   'clutter':     [50, 50, 50]}
    cmap = [*class2color.values()]
    gravity_dim = 2
    """S3DIS dataset, loading the subsampled entire room as input without block/sphere subsampling.
    number of points per room in average, median, and std: (794855.5, 1005913.0147058824, 939501.4733064277)
    Args:
        data_root (str, optional): Defaults to 'data/S3DIS/s3disfull'.
        test_area (int, optional): Defaults to 5.
        voxel_size (float, optional): the voxel size for donwampling. Defaults to 0.04.
        voxel_max (_type_, optional): subsample the max number of point per point cloud. Set None to use all points.  Defaults to None.
        split (str, optional): Defaults to 'train'.
        transform (_type_, optional): Defaults to None.
        loop (int, optional): split loops for each epoch. Defaults to 1.
        presample (bool, optional): wheter to downsample each point cloud before training. Set to False to downsample on-the-fly. Defaults to False.
        variable (bool, optional): where to use the original number of points. The number of point per point cloud is variable. Defaults to False.
    """
    def __init__(self,
                 data_root: str = 'data/S3DIS/s3disfull',
                 test_area: int = 5,
                 voxel_size: float = 0.04,
                 voxel_max=None,
                 split: str = 'train',
                 transform=None,
                 loop: int = 1,
                 presample: bool = False,
                 variable: bool = False,
                 shuffle: bool = True,
                 ):

        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.loop = \
            split, voxel_size, transform, voxel_max, loop
        self.presample = presample
        self.variable = variable
        self.shuffle = shuffle


        self.strides = [4, 4, 4, 4]
        self.k = 32
        self.alpha = 0.1
        self.gs_opts = GaussianOptions.default()
        self.gs_opts.n_cam = 64
        self.gs_opts.fovy = 120
        if self.alpha <= 0:
            self.gs_opts.n_cam = 1

        raw_root = os.path.join(data_root, 'raw')
        self.raw_root = raw_root
        data_list = sorted(os.listdir(raw_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if split == 'train':
            self.data_list = [
                item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [
                item for item in data_list if 'Area_{}'.format(test_area) in item]

        processed_root = os.path.join(data_root, 'processed')
        filename = os.path.join(
            processed_root, f's3dis_{split}_area{test_area}_{voxel_size:.3f}_{str(voxel_max)}.pkl')
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data = []
            for item in tqdm(self.data_list, desc=f'Loading S3DISFull {split} split on Test Area {test_area}'):
                data_path = os.path.join(raw_root, item + '.npy')
                cdata = np.load(data_path).astype(np.float32)
                cdata[:, :3] -= np.min(cdata[:, :3], 0)
                if voxel_size:
                    coord, feat, label = cdata[:,0:3], cdata[:, 3:6], cdata[:, 6:7]
                    uniq_idx = voxelize(coord, voxel_size)
                    coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
                    cdata = np.hstack((coord, feat, label))
                self.data.append(cdata)
            npoints = np.array([len(data) for data in self.data])
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(processed_root, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.data, f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
                print(f"{filename} load successfully")
        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0
        logging.info(f"\nTotally {len(self.data_idx)} samples in {split} set")

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        if self.presample:
            coord, feat, label = np.split(self.data[data_idx], [3, 6], axis=1)
        else:
            data_path = os.path.join(
                self.raw_root, self.data_list[data_idx] + '.npy')
            cdata = np.load(data_path).astype(np.float32)
            cdata[:, :3] -= np.min(cdata[:, :3], 0)
            coord, feat, label = cdata[:, :3], cdata[:, 3:6], cdata[:, 6:7]
            coord, feat, label = crop_pc(
                coord, feat, label, self.split, self.voxel_size, self.voxel_max,
                downsample=not self.presample, variable=self.variable, shuffle=self.shuffle)
            # TODO: do we need to -np.min in cropped data?
        label = label.squeeze(-1).astype(np.int64)
        data = {'pos': coord, 'x': feat, 'y': label}
        # pre-process.
        if self.transform is not None:
            data = self.transform(data)

        if 'heights' not in data.keys():
            data['heights'] = torch.from_numpy(coord[:, self.gravity_dim:self.gravity_dim+1].astype(np.float32))
        data['x'] = torch.cat([data['x'], data['heights']], dim=-1).transpose(0, 1).contiguous().float()
        return self.make_idx(idx, data)

    def __len__(self):
        return len(self.data_idx) * self.loop
        # return 1   # debug

    def make_idx(self, idx, data):
        coord = data['pos'].float()
        gs = NaiveGaussian3D(self.gs_opts, batch_size=8, device=coord.device)
        gs.projects(coord, cam_seed=idx, cam_batch=gs.opt.n_cameras * 2)
        visible = gs.gs_points.visible.squeeze(1).float()
        n_samples = []  # [6000, 1500, 375, 93]
        pre_n = self.voxel_max if self.voxel_max is not None else coord.shape[0]
        for s in self.strides:
            n = pre_n//s
            n_samples.append(n)
            pre_n = n
        # estimating a distance in Euclidean space as the scaler by random fps
        ps, _ = fps_sample(coord.unsqueeze(0), 2, random_start_point=True)
        ps = ps.squeeze(0)
        p0, p1 = ps[0], ps[1]
        scaler = math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2)
        idx_ds = []  # [0, 1, 2, 3],  p
        idx_group_sa = []  # [0, 1, 2, 3], p - sampled_p
        idx_group_la = []  # [0, 1, 2, 3], sampled_p - sampled_p
        for n in n_samples:
            k = self.k
            sub_coord, ds_idx = fps_sample(coord.unsqueeze(0), n)
            sub_coord = sub_coord.squeeze(0)
            ds_idx = ds_idx.squeeze(0)
            sub_visible = visible[ds_idx]
            idx_ds.append(ds_idx)

            kdt_sa = KDTree(coord.numpy(), visible.numpy())
            _, idx_sa = kdt_sa.query(sub_coord.numpy(), sub_visible.numpy(), k=k, alpha=self.alpha, scaler=scaler)
            idx_group_sa.append(torch.from_numpy(idx_sa).int())

            kdt_la = KDTree(sub_coord.numpy(), sub_visible.numpy())
            _, idx_la = kdt_la.query(sub_coord.numpy(), sub_visible.numpy(), k=k, alpha=self.alpha, scaler=scaler)
            idx_group_la.append(torch.from_numpy(idx_la).int())

            coord = sub_coord
            visible = sub_visible
        data['idx_ds'] = idx_ds
        data['idx_group_sa'] = idx_group_sa
        data['idx_group_la'] = idx_group_la
        return data

"""debug
from openpoints.dataset import vis_multi_points
import copy
old_data = copy.deepcopy(data)
if self.transform is not None:
    data = self.transform(data)
vis_multi_points([old_data['pos'][:, :3], data['pos'][:, :3].numpy()], colors=[old_data['x'][:, :3]/255.,data['x'][:, :3].numpy()])
"""
