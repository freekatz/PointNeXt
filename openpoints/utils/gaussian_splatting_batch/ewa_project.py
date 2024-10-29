import torch
from einops import repeat


def ewa_project(
    xyz: torch.Tensor,
    cov3d: torch.Tensor,
    intr: torch.Tensor,
    extr: torch.Tensor,
    uv: torch.Tensor,
    W: int,
    H: int,
    visible: torch.Tensor = None,
    block_width: int = 16
):
    """Project 3D Gaussians to 2D planar Gaussian through elliptical weighted average(EWA).
        a pytorch implementation support batch @msplat

    Args:
        xyz (torch.Tensor): points, [B, N, 3], todo support [N, 3]
        cov3d (torch.Tensor): camera intrinsics, [B, N, 3, 3]
        intr (torch.Tensor): camera intrinsics, [B, 4]
        extr (torch.Tensor): camera pose, [B, 4, 4]
        uv (torch.Tensor): 2D positions for each point in the image, [B, N, 2]
        W (int): view width of camera field size.
        H (int): view height of camera field size.
        visible (torch.Tensor, optional): the visibility status of each point, [B, N, 1], by default None
        block_width (int): side length of tiles inside projection in 2D (always square),
            must be between 2 and 16 inclusive. by default 16

    Returns
        conic (torch.Tensor): the upper-right corner of the 2D covariance matrices, [B, N, 3]
        radius (torch.Tensor): radius of the 2D planar Gaussian on the image, [B, N, 1]
        tiles (torch.Tensor): number of tiles covered by 2D planar Gaussians on the image, [B, N, 1]
    """
    device = xyz.device
    block_height = block_width
    B, N, _ = xyz.shape
    if visible is None:
        visible = torch.ones((B, N, 1), device=xyz.device)
    else:
        visible = visible.sum(dim=-1, keepdim=False)
        visible = (visible > 0).int()
    visible = visible.squeeze(dim=-1)

    # use the upper-right corner of cov3d
    cov3d = torch.stack([cov3d[:, :, 0, 0], cov3d[:, :, 0, 1],
                         cov3d[:, :, 0, 2], cov3d[:, :, 1, 1],
                         cov3d[:, :, 1, 2], cov3d[:, :, 2, 2]], dim=-1)  # [B, N, 6]

    Wmat = extr[..., :3, :3]
    p = extr[..., :3, 3]
    Wmat = repeat(Wmat, 'b m d -> b n m d', n=xyz.shape[1])
    p = repeat(p, 'b d -> b n d', n=xyz.shape[1])

    t = torch.matmul(Wmat, xyz[:, ..., None])[:, ..., 0] + p
    rz = 1.0 / t[:, ..., 2]
    rz2 = rz ** 2

    fx = intr[:, ..., 0].unsqueeze(-1)
    fy = intr[:, ..., 1].unsqueeze(-1)
    Jmat = torch.stack(
        [
            torch.stack([fx * rz, torch.zeros_like(rz), -fx * t[:, ..., 0] * rz2], dim=-1),
            torch.stack([torch.zeros_like(rz), fy * rz, -fy * t[:, ..., 1] * rz2], dim=-1),
        ],
        dim=-2,
    )
    T = Jmat @ Wmat
    cov3d_1 = torch.stack([cov3d[:, ..., 0], cov3d[:, ..., 1], cov3d[:, ..., 2]], dim=-1)
    cov3d_2 = torch.stack([cov3d[:, ..., 1], cov3d[:, ..., 3], cov3d[:, ..., 4]], dim=-1)
    cov3d_3 = torch.stack([cov3d[:, ..., 2], cov3d[:, ..., 4], cov3d[:, ..., 5]], dim=-1)
    cov3d = torch.stack([cov3d_1, cov3d_2, cov3d_3], dim=-1)

    cov2d = T @ cov3d @ T.transpose(-1, -2)
    cov2d[:, ..., 0, 0] = cov2d[:, :, 0, 0] + 0.3
    cov2d[:, ..., 1, 1] = cov2d[:, :, 1, 1] + 0.3

    # Compute extent in screen space
    det = cov2d[:, ..., 0, 0] * cov2d[:, ..., 1, 1] - cov2d[:, ..., 0, 1] ** 2
    det_mask = det != 0

    conic = torch.stack(
        [
            cov2d[:, ..., 1, 1] / det,
            -cov2d[:, ..., 0, 1] / det,
            cov2d[:, ..., 0, 0] / det,
        ],
        dim=-1,
    )

    b = (cov2d[:, ..., 0, 0] + cov2d[:, ..., 1, 1]) / 2
    v1 = b + torch.sqrt(torch.clamp(b ** 2 - det, min=0.1))
    v2 = b - torch.sqrt(torch.clamp(b ** 2 - det, min=0.1))
    radius = torch.ceil(3.0 * torch.sqrt(torch.max(v1, v2)))

    # get tiles
    top_left = torch.zeros_like(uv, dtype=torch.int, device=device)
    bottom_right = torch.zeros_like(uv, dtype=torch.int, device=device)
    top_left[:, :, 0] = ((uv[:, :, 0] - radius) / block_width)
    top_left[:, :, 1] = ((uv[:, :, 1] - radius) / block_height)
    bottom_right[:, :, 0] = ((uv[:, :, 0] + radius + block_width - 1) / block_width)
    bottom_right[:, :, 1] = ((uv[:, :, 1] + radius + block_height - 1) / block_height)

    tile_bounds = torch.zeros(2, dtype=torch.int, device=device)
    tile_bounds[0] = (W + block_width - 1) / block_width
    tile_bounds[1] = (H + block_height - 1) / block_height

    tile_min = torch.stack(
        [
            torch.clamp(top_left[:, ..., 0], 0, tile_bounds[0]),
            torch.clamp(top_left[:, ..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    tile_max = torch.stack(
        [
            torch.clamp(bottom_right[:, ..., 0], 0, tile_bounds[0]),
            torch.clamp(bottom_right[:, ..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )

    tiles_tmp = tile_max - tile_min
    tiles = tiles_tmp[:, ..., 0] * tiles_tmp[:, ..., 1]

    mask = torch.logical_and(tiles != 0, det_mask)
    if visible is not None:
        mask = torch.logical_and(visible.squeeze(), mask)

    conic = torch.nan_to_num(conic)
    radius = torch.nan_to_num(radius)
    tiles = torch.nan_to_num(tiles)

    conic = conic * mask.float()[:, ..., None]
    radius = radius * mask.float()
    tiles = tiles * mask.float()

    return conic, radius.int().unsqueeze(-1), tiles.int().unsqueeze(-1)


if __name__ == '__main__':
    xyz = torch.randn((1, 20, 3))
    cov3d = torch.randn((1, 20, 3, 3))
    intr = torch.randn((1, 4))
    extr = torch.randn((1, 4, 4))
    uv = torch.randn((1, 20, 2))
    visible = torch.randn((1, 20, 1))

    cov2d, _, _ = ewa_project(xyz, cov3d, intr, extr, uv, 512, 512, None)
    print(cov2d.shape, cov2d)
