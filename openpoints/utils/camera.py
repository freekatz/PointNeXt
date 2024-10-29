import torch
from kiui.op import safe_normalize
from kiui.cam import undo_orbit_camera


def look_at(campos, target, opengl=True, device='cuda'):
    """construct pose rotation matrix by look-at. a pytorch implementation @kiui

    Args:
        campos (torch.Tensor): camera position, float [3]
        target (torch.Tensor): look at target, float [3]
        opengl (bool, optional): whether use opengl camera convention (forward direction is target --> camera). by default True.
        device (str, optional): device. by default torch.device('cuda')

    Returns:
        torch.Tensor: the camera pose rotation matrix, float [3, 3], normalized.
    """
    if not opengl:
        # forward is camera --> target
        forward_vector = safe_normalize(target - campos)
        up_vector = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
        right_vector = safe_normalize(torch.cross(forward_vector, up_vector))
        up_vector = safe_normalize(torch.cross(right_vector, forward_vector))
    else:
        # forward is target --> camera
        forward_vector = safe_normalize(campos - target)
        up_vector = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
        right_vector = safe_normalize(torch.cross(up_vector, forward_vector, dim=-1))
        up_vector = safe_normalize(torch.cross(forward_vector, right_vector, dim=-1))
    R = torch.stack([right_vector, up_vector, forward_vector], dim=1)
    return R


class OrbitCamera:
    """ An orbital camera class. a custom pytorch implementation @kiui
    """

    def __init__(self, camid: int, width: int, height: int, campos: tuple, target: tuple = None,
                 fovy: float = 60, cam_index = -1, target_index = -1, device='cuda'):
        """init function

        Args:
            camid (int): camera id.
            width (int): view width of camera field size.
            height (int): view height of camera field size.
            campos (tuple): camera position.
            target (tuple, optional): look at target position. by default (0, 0, 0).
            fovy (float, optional): camera field of view in degree along y-axis. by default 60.
            cam_index (float, optional): the camera index in point cloud. by default -1.
            device (str, optional): device. by default 'cuda'
        """
        self.camid = torch.tensor(camid, device=device)
        self.W = width
        self.H = height
        self.campos = torch.tensor(campos, dtype=torch.float32, device=device)
        self.fovy = torch.deg2rad(torch.tensor([fovy], dtype=torch.float32, device=device))[0]
        if target is None:
            target = (0, 0, 0)
        self.target = torch.tensor(target, dtype=torch.float32, device=device)  # look at this point
        self.cam_index = cam_index
        self.target_index = target_index
        self.device = device

        elevation, azimuth, radius = undo_orbit_camera(self.pose.detach().cpu().numpy())
        self.elevation = elevation
        self.azimuth = azimuth
        self.radius = radius

    @property
    def fovx(self):
        return 2 * torch.arctan(torch.tan(self.fovy / 2) * self.W / self.H)

    @property
    def pose(self):
        pose = torch.eye(4, dtype=torch.float32, device=self.device)
        pose[:3, :3] = look_at(self.campos, self.target, device=self.device)
        pose[:3, 3] = self.campos
        return pose

    @property
    def intrinsics(self):
        focal = self.H / (2 * torch.tan(self.fovy / 2))
        return torch.tensor([focal, focal, self.W // 2, self.H // 2], dtype=torch.float32, device=self.device)
