import math
import random
from dataclasses import dataclass, field

import torch
from einops import repeat
from torch import nn

from .point_utils import points_centroid, points_scaler
from .camera import OrbitCamera
from .gaussian_splatting_batch import project_points, compute_cov3d, ewa_project
from pykdtree.kdtree import KDTree
from .cutils import grid_subsampling
from .subsample import create_sampler, fps_sample, visible_sample


@dataclass
class GaussianOptions(dict):
    # camera numbers outside in points
    n_cameras: int = 4
    # camera field of view in degree along y-axis.
    cam_fovy: float = 60.0
    # camera field size, [width, height]
    cam_field_size: list = field(default_factory=list)
    # camera sampler
    cam_sampler: str = 'fps'
    # generate camera method
    cam_gen_method: str = 'centroid'  # ['centroid', 'farthest']

    @classmethod
    def default(cls):
        return GaussianOptions(
            n_cameras=4,
            cam_fovy=60.0,
            cam_field_size=[512, 512],
            cam_sampler='fps',
            cam_gen_method='centroid'
        )

    def __str__(self):
        return f'''GaussianOptions(
            n_cameras={self.n_cameras},
            cam_fovy={self.cam_fovy},
            cam_field_size={self.cam_field_size},
            cam_sampler={self.cam_sampler},
            cam_gen_method={self.cam_gen_method})'''


class GaussianPoints(object):
    def __set_attr__(self, key, value):
        assert not self.__is_attr_exists__(key)
        self.__dict__[key] = value

    def __update_attr__(self, key, value):
        self.__dict__[key] = value

    def __get_attr__(self, key):
        if not self.__is_attr_exists__(key):
            return None
        return self.__dict__[key]

    def __del_attr__(self, key):
        if not self.__is_attr_exists__(key):
            return
        self.__dict__.pop(key)

    def __is_attr_exists__(self, key):
        return key in self.__dict__.keys()

    def keys(self):
        return self.__dict__.keys()

    def to_cuda(self, non_blocking=True):
        keys = self.keys()
        for key in keys:
            item = self.__get_attr__(key)
            if isinstance(item, torch.Tensor):
                item = item.cuda(non_blocking=non_blocking)
            if isinstance(item, list):
                for i in range(len(item)):
                    if isinstance(item[i], torch.Tensor):
                        item[i] = item[i].cuda(non_blocking=non_blocking)
            self.__update_attr__(key, item)

    @property
    def layer_idx(self):
        return self.__get_attr__('layer_idx')

    @property
    def idx_ds(self):
        return self.__get_attr__('idx_ds')

    @property
    def idx_us(self):
        return self.__get_attr__('idx_us')

    @property
    def idx_group(self):
        return self.__get_attr__('idx_group')

    @property
    def pts_list(self):
        return self.__get_attr__('pts_list')

    @property
    def p(self):
        return self.__get_attr__('p')

    @property
    def f(self):
        return self.__get_attr__('f')

    @property
    def y(self):
        return self.__get_attr__('y')

    @property
    def f_gs(self, layer_idx=None):
        # [uv, depths, visible, camid], [N, 2+1+1+1, n_cameras*2]
        item = self.__get_attr__('f_gs')
        if layer_idx is not None:
            ds_idx = self.idx_ds[layer_idx]
            item = item[ds_idx]
            self.__update_attr__('f_gs', item)
        return item

    @property
    def cameras(self):
        cameras = self.__get_attr__('cameras')
        return cameras

    @property
    def uv(self, layer_idx=None):
        item = self.__get_attr__('uv')
        if layer_idx is not None:
            ds_idx = self.idx_ds[layer_idx]
            item = item[ds_idx]
            self.__update_attr__('uv', item)
        return item

    @property
    def depths(self, layer_idx=None):
        item = self.__get_attr__('depths')
        if layer_idx is not None:
            ds_idx = self.idx_ds[layer_idx]
            item = item[ds_idx]
            self.__update_attr__('depths', item)
        return item

    @property
    def visible(self, layer_idx=None):
        item = self.__get_attr__('visible')
        if layer_idx is not None:
            ds_idx = self.idx_ds[layer_idx]
            item = item[ds_idx]
            self.__update_attr__('visible', item)
        return item

    @property
    def camid(self, layer_idx=None):
        item = self.__get_attr__('camid')
        if layer_idx is not None:
            ds_idx = self.idx_ds[layer_idx]
            item = item[ds_idx]
            self.__update_attr__('camid', item)
        return item

    @property
    def cam_intr(self):
        return self.__get_attr__('cam_intr')

    @property
    def cam_extr(self):
        return self.__get_attr__('cam_extr')

    @property
    def cov3d(self):
        return self.__get_attr__('cov3d')

    @property
    def cov2d(self):
        return self.__get_attr__('cov2d')


class NaiveGaussian3D:
    """
    A pipeline about 3D naive gaussian to process the point cloud
    """

    def __init__(self,
                 opt: GaussianOptions = None,
                 batch_size: int = 8,
                 device: str = 'cuda',
                 **kwargs):
        if opt is None:
            opt = GaussianOptions.default()
        self.opt = opt
        self.batch_size = batch_size
        self.device = device
        self.cam_sampler = create_sampler(self.opt.cam_sampler)

        self.gs_points = GaussianPoints()

    def init_points(self):
        self.gs_points = GaussianPoints()

    def to(self, device):
        self.device = device
        return self

    def generate_cameras(self, xyz):
        """
        :param xyz: [N, 3]
        :return: [n_cameras*2]
        """
        gen_method = self.opt.cam_gen_method
        if gen_method == 'centroid':
            return self.generate_cameras_by_centroid(xyz)
        elif gen_method == 'farthest':
            return self.generate_cameras_by_farthest(xyz)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def generate_cameras_by_centroid(self, xyz, shuffle=False):
        """
        :param xyz: [N, 3]
        :param shuffle: shuffle the cameras
        :return: [n_cameras*2]
        """
        n_cameras = self.opt.n_cameras
        cam_fovy = self.opt.cam_fovy
        cam_width, cam_height = self.opt.cam_field_size

        centroid = points_centroid(xyz.unsqueeze(0)).squeeze(0)
        xyz_sampled, _ = self.cam_sampler(xyz=xyz.unsqueeze(0), n_samples=n_cameras)
        xyz_sampled = xyz_sampled.squeeze(0)
        cameras_all = []
        for j in range(n_cameras):
            cx, cy, cz = centroid
            x, y, z = xyz_sampled[j]
            cameras_all.append(OrbitCamera(
                camid=2 * j + 1,
                width=cam_width,
                height=cam_height,
                campos=(x, y, z),
                target=(cx, cy, cz),
                fovy=cam_fovy,
                device=self.device,
            ))
            cameras_all.append(OrbitCamera(
                camid=2 * j + 2,
                width=cam_width,
                height=cam_height,
                campos=(cx, cy, cz),
                target=(x, y, z),
                fovy=cam_fovy,
                device=self.device,
            ))
        if shuffle:
            random.shuffle(cameras_all)
        self.gs_points.__update_attr__('cameras', cameras_all)
        return cameras_all

    @torch.no_grad()
    def generate_cameras_by_farthest(self, xyz):
        """
        :param xyz: [N, 3]
        :return: [n_cameras*2]
        """
        n_cameras = self.opt.n_cameras
        cam_fovy = self.opt.cam_fovy
        cam_width, cam_height = self.opt.cam_field_size
        assert self.opt.cam_sampler == 'fps'

        xyz_sampled, idx = self.cam_sampler(xyz=xyz.unsqueeze(0), n_samples=n_cameras * 2 + 1)
        xyz_sampled = xyz_sampled.squeeze(0)
        cameras_all = []
        for j in range(1, n_cameras * 2 + 1):
            cx, cy, cz = xyz_sampled[j - 1]
            x, y, z = xyz_sampled[j]
            cameras_all.append(OrbitCamera(
                camid=j,
                width=cam_width,
                height=cam_height,
                campos=(x, y, z),
                target=(cx, cy, cz),
                fovy=cam_fovy,
                cam_index=idx[0][j],
                target_index=idx[0][j-1],
                device=self.device,
            ))
        self.gs_points.__update_attr__('cameras', cameras_all)
        return cameras_all

    @torch.no_grad()
    def projects(self, xyz, cam_seed=0, scale=1., cam_batch=1):
        """
        :param xyz: [N, 3]
        :param cam_seed: seed to generate camera id
        :param scale: xyz scale factor
        :param cam_batch: batch size of cameras to project
        :return: [N, 3]
        """
        assert len(xyz.shape) == 2
        cam_seed = cam_seed % self.batch_size
        n_cameras = self.opt.n_cameras
        assert n_cameras * 2 % cam_batch == 0
        cam_width, cam_height = self.opt.cam_field_size
        xyz_scaled = points_scaler(xyz.unsqueeze(0), scale=scale).squeeze(0)
        cameras = self.generate_cameras(xyz_scaled)

        uv_all, depths_all, visible_all = [], [], []
        cam_intr_all, cam_extr_all = [], []
        camid = torch.zeros((n_cameras*2, xyz_scaled.shape[0], 1), device=self.device)
        for j in range(n_cameras * 2 // cam_batch):
            cam_intr_batch = []
            cam_extr_batch = []
            for i in range(cam_batch):
                cam_intr = cameras[j*cam_batch + i].intrinsics
                cam_extr = cameras[j*cam_batch + i].pose
                camid[j*cam_batch + i, ...] = cameras[j*cam_batch + i].camid + cam_seed * n_cameras * 2
                cam_intr_batch.append(cam_intr)
                cam_extr_batch.append(cam_extr)
            cam_intr = torch.stack(cam_intr_batch, dim=0)
            cam_extr = torch.stack(cam_extr_batch, dim=0)
            uv, depths = project_points(
                # the batch size is cam_batch
                xyz_scaled,
                cam_intr,
                cam_extr,
                cam_width,
                cam_height,
            )
            visible = (depths != 0).int()

            uv_all.append(uv)
            depths_all.append(depths)
            visible_all.append(visible)
            cam_intr_all.append(cam_intr)
            cam_extr_all.append(cam_extr)

        uv = torch.cat(uv_all, dim=0).permute(1, 2, 0)
        depths = torch.cat(depths_all, dim=0).permute(1, 2, 0)
        visible = torch.cat(visible_all, dim=0).permute(1, 2, 0)
        camid = camid.permute(1, 2, 0)
        cam_intr = torch.cat(cam_intr_all, dim=0).permute(1, 0)
        cam_extr = torch.cat(cam_extr_all, dim=0).permute(1, 2, 0)

        self.gs_points.__update_attr__('uv', uv)
        self.gs_points.__update_attr__('depths', depths)
        self.gs_points.__update_attr__('visible', visible)
        self.gs_points.__update_attr__('camid', camid)
        self.gs_points.__update_attr__('cam_intr', cam_intr)
        self.gs_points.__update_attr__('cam_extr', cam_extr)
        # f_gs = torch.cat(([uv, depths, visible, camid]), dim=1)
        self.gs_points.__update_attr__('f_gs', visible.squeeze(1).float())

    @torch.no_grad()
    def cov3d(self, xyz_padded):
        # todo support manual idx
        """
        :param xyz_padded: [N, n_neighbors, 3]
        :return: [N, 3, 3]
        """
        visible = self.gs_points.visible
        cov3d = compute_cov3d(xyz_padded.unsqueeze(0), visible.unsqueeze(0)).squeeze(0)
        return cov3d

    @torch.no_grad()
    def cov2d(self, xyz, cov3d):
        # todo support manual idx
        """
        :param xyz: [N, 3]
        :param cov3d: [N, 3, 3]
        :return: [N, 3, n_cameras*2]
        """
        n_cameras = self.opt.n_cameras
        cam_width, cam_height = self.opt.cam_field_size
        uv = self.gs_points.uv.unsqueeze(0)
        visible = self.gs_points.visible.unsqueeze(0)
        cam_intr = self.gs_points.cam_intr.unsqueeze(0)
        cam_extr = self.gs_points.cam_extr.unsqueeze(0)
        xyz = xyz.unsqueeze(0)
        cov3d = cov3d.unsqueeze(0)

        cov2d_all = []
        for j in range(n_cameras * 2):
            cov2d, _, _ = ewa_project(
                xyz=xyz,
                cov3d=cov3d,
                intr=cam_intr[:, :, j].squeeze(-1),
                extr=cam_extr[:, :, :, j].squeeze(-1),
                uv=uv[:, :, :, j].squeeze(-1),
                W=cam_width,
                H=cam_height,
                visible=visible[:, :, :, j].squeeze(-1),
            )
            cov2d_all.append(cov2d)
        cov2d = torch.stack(cov2d_all, dim=-1).squeeze(0)
        return cov2d


def make_gs_points(gs_points, ks, grid_size=None, n_samples=None, up_sample=True, visible_sample_stride=0., alpha=0.) -> GaussianPoints:
    assert (grid_size is not None and n_samples is not None) is False
    assert (grid_size is None and n_samples is None) is False
    n_layers = len(ks)
    full_p = gs_points.p
    full_visible = gs_points.visible.squeeze(1).float()

    # estimating a distance in Euclidean space as the scaler by random fps
    ps, _ = fps_sample(full_p.unsqueeze(0), 2, random_start_point=True)
    ps = ps.squeeze(0)
    p0, p1 = ps[0], ps[1]
    scaler = math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2)

    full_p = full_p.contiguous()
    full_visible = full_visible.contiguous()
    visible = full_visible
    p = full_p

    idx_ds = []
    idx_us = []
    idx_group = []
    idx_gs_group = []
    for i in range(n_layers):
        # down sample
        if i > 0:
            if grid_size is not None:
                gsize = grid_size[i-1]
                if p.is_cuda:
                    ds_idx = grid_subsampling(p.detach().cpu(), gsize)
                else:
                    ds_idx = grid_subsampling(p, gsize)
            else:
                if visible_sample_stride > 0 and i == 1:
                    _, ds_idx = visible_sample(p.unsqueeze(0), gs_points.visible.unsqueeze(0), int(p.shape[0] // visible_sample_stride))
                    ds_idx = ds_idx.squeeze(0)
                else:
                    _, ds_idx = fps_sample(p.unsqueeze(0), n_samples[i-1])
                    ds_idx = ds_idx.squeeze(0)
            p = p[ds_idx]
            visible = visible[ds_idx]
            idx_ds.append(ds_idx)

        # group
        k = ks[i]
        kdt = KDTree(p.numpy(), visible.numpy())
        _, idx = kdt.query(p.numpy(), visible.numpy(), k=k, alpha=alpha, scaler=scaler)
        idx_group.append(torch.from_numpy(idx).long())

        # up sample
        if i > 0 and up_sample:
            _, us_idx = kdt.query(full_p.numpy(), full_visible.numpy(), k=1, alpha=alpha, scaler=scaler)
            idx_us.append(torch.from_numpy(us_idx).long())

    gs_points.__update_attr__('idx_ds', idx_ds)
    gs_points.__update_attr__('idx_us', idx_us)
    gs_points.__update_attr__('idx_group', idx_group)
    gs_points.__update_attr__('idx_gs_group', idx_gs_group)
    return gs_points


def make_gs_features(gs: NaiveGaussian3D):
    uv = gs.gs_points.uv
    depths = gs.gs_points.depths
    group_idx = gs.gs_points.idx_group[0]
    cov3d = gs.cov3d(gs.gs_points.p[group_idx])
    cov2d = gs.cov2d(gs.gs_points.p, cov3d)
    # gs feature
    delta = torch.round((uv % 1) * 1e5) / 1e5  # [N, 2, n_cameras*2]
    power = -(
            0.5 * cov2d[:, 0, :] * delta[:, 0, :] * delta[:, 0, :]
            + 0.5 * cov2d[:, 2, :] * delta[:, 1, :] * delta[:, 1, :]
            + cov2d[:, 1, :] * delta[:, 0, :] * delta[:, 1, :])  # [N, n_cameras*2]
    # use depths as opacity
    opacity = nn.functional.softmax(depths.squeeze(1), dim=0)  # [N, n_cameras*2]
    alpha = torch.clamp(torch.exp(power), min=1.0 / 255.0, max=0.99)  # [N, n_cameras*2]
    colors = opacity * alpha * (1 - alpha)  # [N, n_cameras*2]
    colors = torch.stack([colors[:, 0], colors[:, 1:].max(1)[0] + colors[:, 1:].mean(1)[0]], dim=-1)
    return colors


def merge_gs_list(gs_list, up_sample=True) -> NaiveGaussian3D:
    assert len(gs_list) > 0
    new_gs = NaiveGaussian3D(gs_list[0].opt, batch_size=len(gs_list))

    p_all = []
    f_gs_all = []
    f_all = []
    y_all = []
    idx_ds_all = []
    idx_us_all = []
    idx_group_all = []
    pts_all = []
    n_layers = len(gs_list[0].gs_points.idx_group)
    pts_per_layer = [0] * n_layers
    for i in range(len(gs_list)):
        gs = gs_list[i]
        p_all.append(gs.gs_points.p)
        f_gs_all.append(gs.gs_points.f_gs)
        f_all.append(gs.gs_points.f)
        y_all.append(gs.gs_points.y)

        idx_ds = gs.gs_points.idx_ds
        idx_us = gs.gs_points.idx_us
        idx_group = gs.gs_points.idx_group
        pts = []
        for layer_idx in range(n_layers):
            if layer_idx < len(idx_ds):
                idx_ds[layer_idx].add_(pts_per_layer[layer_idx])
                if up_sample:
                    idx_us[layer_idx].add_(pts_per_layer[layer_idx + 1])
            idx_group[layer_idx].add_(pts_per_layer[layer_idx])
            pts.append(idx_group[layer_idx].shape[0])
        idx_ds_all.append(idx_ds)
        idx_us_all.append(idx_us)
        idx_group_all.append(idx_group)
        pts_all.append(pts)
        pts_per_layer = [pt + idx.shape[0] for (pt, idx) in zip(pts_per_layer, idx_group)]

    p = torch.cat(p_all, dim=0)
    new_gs.gs_points.__update_attr__('p', p)

    f_gs = torch.cat(f_gs_all, dim=0)
    new_gs.gs_points.__update_attr__('f_gs', f_gs)

    f = torch.cat(f_all, dim=0)
    new_gs.gs_points.__update_attr__('f', f)

    y = torch.cat(y_all, dim=0)
    new_gs.gs_points.__update_attr__('y', y)

    idx_ds = [torch.cat(idx, dim=0) for idx in zip(*idx_ds_all)]
    new_gs.gs_points.__update_attr__('idx_ds', idx_ds)  # layer_idx: [1, 2, 3]

    idx_us = [torch.cat(idx, dim=0) for idx in zip(*idx_us_all)]
    new_gs.gs_points.__update_attr__('idx_us', idx_us)  # layer_idx: [2, 1, 0]

    idx_group = [torch.cat(idx, dim=0) for idx in zip(*idx_group_all)]
    new_gs.gs_points.__update_attr__('idx_group', idx_group)  # layer_idx: [0, 1, 2, 3]

    pts_list = torch.tensor(pts_all, dtype=torch.int64)
    pts_list = pts_list.view(-1, n_layers).transpose(0, 1).contiguous()  # batch_size * layer_idx: [0, 1, 2, 3]
    new_gs.gs_points.__update_attr__('pts_list', pts_list)
    return new_gs


if __name__ == '__main__':
    xyz = torch.randn((20, 3))
    gs = NaiveGaussian3D(
        opt=GaussianOptions.default(),
        batch_size=1,
    )
    gs.projects(xyz)
    gs.gs_points = make_gs_points(
        gs.gs_points,
        [4],
        [4],
        None,
        [1, 2]
    )
    colors = make_gs_features(gs)
    print(colors.shape)