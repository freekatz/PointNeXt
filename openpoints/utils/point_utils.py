import torch


def points_centroid(xyz):
    """
    :param xyz: [B, N, 3]
    :return: [B, 3]
    """
    return torch.mean(xyz, dim=1)


def points_scaler(xyz, scale=1.):
    """
    :param xyz: [B, N, 3]
    :param scale: float, scale factor, by default 1.0, which means scale into [0, 1]
    :return: [B, N, 3]]
    """
    mi, ma = xyz.min(dim=1, keepdim=True)[0], xyz.max(dim=1, keepdim=True)[0]
    xyz = (xyz - mi) / (ma - mi + 1e-12)
    return xyz * scale

