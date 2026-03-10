"""
Rotation and transformation utilities for 3D geometry.

All functions use PyTorch tensors and support batched operations.
Quaternion convention: [w, x, y, z] (scalar first).
"""

import torch

# =============================================================================
# Quaternion Operations
# =============================================================================

def quat_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion tensor of shape (batch, 4) with [w, x, y, z] convention
        
    Returns:
        Rotation matrix of shape (batch, 3, 3)
    """
    # Ensure input q is a torch.Tensor
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q)
    batch_size = q.shape[0]

    if q.ndim == 1:
        q = q.unsqueeze(0)

    # Normalize quaternion
    q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros((batch_size, 3, 3), device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    return R


def rotation_matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion.
    
    Args:
        R: Rotation matrix of shape (batch, 3, 3)
        
    Returns:
        Quaternion tensor of shape (batch, 4) with [w, x, y, z] convention
    """
    batch_size = R.shape[0]
    device = R.device
    dtype = R.dtype

    q = torch.zeros((batch_size, 4), device=device, dtype=dtype)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    for i in range(batch_size):
        if trace[i] > 0:
            s = 0.5 / torch.sqrt(trace[i] + 1.0)
            q[i, 0] = 0.25 / s
            q[i, 1] = (R[i, 2, 1] - R[i, 1, 2]) * s
            q[i, 2] = (R[i, 0, 2] - R[i, 2, 0]) * s
            q[i, 3] = (R[i, 1, 0] - R[i, 0, 1]) * s
        elif R[i, 0, 0] > R[i, 1, 1] and R[i, 0, 0] > R[i, 2, 2]:
            s = 2.0 * torch.sqrt(1.0 + R[i, 0, 0] - R[i, 1, 1] - R[i, 2, 2])
            q[i, 0] = (R[i, 2, 1] - R[i, 1, 2]) / s
            q[i, 1] = 0.25 * s
            q[i, 2] = (R[i, 0, 1] + R[i, 1, 0]) / s
            q[i, 3] = (R[i, 0, 2] + R[i, 2, 0]) / s
        elif R[i, 1, 1] > R[i, 2, 2]:
            s = 2.0 * torch.sqrt(1.0 + R[i, 1, 1] - R[i, 0, 0] - R[i, 2, 2])
            q[i, 0] = (R[i, 0, 2] - R[i, 2, 0]) / s
            q[i, 1] = (R[i, 0, 1] + R[i, 1, 0]) / s
            q[i, 2] = 0.25 * s
            q[i, 3] = (R[i, 1, 2] + R[i, 2, 1]) / s
        else:
            s = 2.0 * torch.sqrt(1.0 + R[i, 2, 2] - R[i, 0, 0] - R[i, 1, 1])
            q[i, 0] = (R[i, 1, 0] - R[i, 0, 1]) / s
            q[i, 1] = (R[i, 0, 2] + R[i, 2, 0]) / s
            q[i, 2] = (R[i, 1, 2] + R[i, 2, 1]) / s
            q[i, 3] = 0.25 * s

    return q


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions: q1 * q2.
    
    Args:
        q1, q2: Quaternion tensors of shape (batch, 4) with [w, x, y, z] convention
        
    Returns:
        Product quaternion of shape (batch, 4)
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion conjugate.
    
    Args:
        q: Quaternion tensor of shape (batch, 4) with [w, x, y, z] convention
        
    Returns:
        Conjugate quaternion [w, -x, -y, -z] of shape (batch, 4)
    """
    return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=1)


# =============================================================================
# Euler Angle Conversions
# =============================================================================

def euler_from_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        q: Quaternion tensor of shape (batch, 4) with [w, x, y, z] convention
        
    Returns:
        Euler angles of shape (batch, 3) as [roll, pitch, yaw]
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return torch.stack([roll, pitch, yaw], dim=1)


def quaternion_from_euler(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles to quaternion.
    
    Args:
        roll, pitch, yaw: Euler angles of shape (batch,)
        
    Returns:
        Quaternion of shape (batch, 4) with [w, x, y, z] convention
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return torch.stack([w, x, y, z], dim=1)


def yaw_from_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from quaternion.
    
    Args:
        q: Quaternion tensor (can be squeezed or batched)
        
    Returns:
        Yaw angle tensor
    """
    w, x, y, z = q.squeeze()
    yaw = torch.arctan2(
        torch.tensor([2.0 * (w*z + x*y)]),
        torch.tensor([1.0 - 2.0 * (y*y + z*z)])
    )
    return yaw


def rotation_matrix_to_euler(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to Euler angles.
    
    Args:
        R: Rotation matrix of shape (batch, 3, 3)
        
    Returns:
        Euler angles of shape (batch, 3) as [roll, pitch, yaw]
    """
    return euler_from_quaternion(rotation_matrix_to_quat(R))


def rotation_matrix_from_euler(euler: torch.Tensor, degrees: bool = False) -> torch.Tensor:
    """Create rotation matrix from Euler angles.
    
    Convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    
    Args:
        euler: Euler angles [roll, pitch, yaw] of shape (3,) or (batch, 3)
        degrees: If True, input angles are in degrees
        
    Returns:
        Rotation matrix of shape (3, 3) or (batch, 3, 3)
    """
    angles = torch.as_tensor(euler)
    
    if degrees:
        angles = angles * (torch.pi / 180.0)

    # Handle unbatched input
    if angles.ndim == 1 and angles.numel() == 3:
        batched = False
        angles = angles.unsqueeze(0)
    else:
        batched = True

    roll = angles[..., 0]
    pitch = angles[..., 1]
    yaw = angles[..., 2]

    cx = torch.cos(roll)
    sx = torch.sin(roll)
    cy = torch.cos(pitch)
    sy = torch.sin(pitch)
    cz = torch.cos(yaw)
    sz = torch.sin(yaw)

    # Rotation matrices (batchable)
    Rx = torch.stack([
        torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=-1),
        torch.stack([torch.zeros_like(cx), cx, -sx], dim=-1),
        torch.stack([torch.zeros_like(cx), sx, cx], dim=-1),
    ], dim=-2)

    Ry = torch.stack([
        torch.stack([cy, torch.zeros_like(cy), sy], dim=-1),
        torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=-1),
        torch.stack([-sy, torch.zeros_like(cy), cy], dim=-1),
    ], dim=-2)

    Rz = torch.stack([
        torch.stack([cz, -sz, torch.zeros_like(cz)], dim=-1),
        torch.stack([sz, cz, torch.zeros_like(cz)], dim=-1),
        torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], dim=-1),
    ], dim=-2)

    # Combined rotation: R = Rz @ Ry @ Rx
    R = Rz.matmul(Ry.matmul(Rx))

    if not batched:
        return R[0]
    return R


# =============================================================================
# SO(3) Operations
# =============================================================================

def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """Create skew-symmetric matrix from vector.
    
    Args:
        v: Vector of shape (3,) or (N, 3)
    
    Returns:
        Skew-symmetric matrix of shape (3, 3) or (N, 3, 3)
    """
    if v.ndim == 1:
        assert v.shape[0] == 3, f"Expected vector of length 3, got {v.shape[0]}"
        return torch.tensor([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ], dtype=v.dtype, device=v.device)
    else:
        assert v.shape[-1] == 3, f"Expected last dimension to be 3, got {v.shape[-1]}"
        zeros = torch.zeros_like(v[..., 0])
        return torch.stack([
            torch.stack([zeros, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], zeros, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], zeros], dim=-1),
        ], dim=-2)


def vee_map(R: torch.Tensor) -> torch.Tensor:
    """Extract vector from skew-symmetric matrix (inverse of skew_symmetric).
    
    Args:
        R: Skew-symmetric matrix of shape (3, 3) or (N, 3, 3)
    
    Returns:
        Vector of shape (3,) or (N, 3)
    """
    if R.ndim == 2:
        assert R.shape == (3, 3), f"Expected (3, 3) matrix, got {R.shape}"
        return torch.tensor([R[2, 1], R[0, 2], R[1, 0]], dtype=R.dtype, device=R.device)
    else:
        assert R.shape[-2:] == (3, 3), f"Expected (..., 3, 3) matrix, got {R.shape}"
        return torch.stack([R[..., 2, 1], R[..., 0, 2], R[..., 1, 0]], dim=-1)


def rotation_error(R: torch.Tensor, R_des: torch.Tensor) -> torch.Tensor:
    """Compute rotation error vector from SO(3) matrices.
    
    The error is computed as the logarithmic map of R_des^T @ R, which gives
    the rotation vector needed to go from R to R_des.
    
    Args:
        R: Current rotation matrix (3, 3) or (N, 3, 3)
        R_des: Desired rotation matrix (3, 3) or (N, 3, 3)
    
    Returns:
        Error vector of shape (3,) or (N, 3)
    """
    # Compute relative rotation: R_error = R_des^T @ R
    if R.ndim == 2:
        assert R.shape == (3, 3), f"Expected (3, 3) rotation matrix, got {R.shape}"
        assert R_des.shape == (3, 3), f"Expected (3, 3) desired rotation matrix, got {R_des.shape}"
        R_error = R_des.T @ R
        
        # Extract rotation angle
        trace = torch.trace(R_error)
        angle = torch.arccos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
        
        if angle.abs() < 1e-6:
            return torch.zeros(3, dtype=R.dtype, device=R.device)
        
        # Extract axis from skew-symmetric part
        axis = vee_map(R_error - R_error.T) / (2 * torch.sin(angle))
        return -axis * angle
    else:
        assert R.shape[-2:] == (3, 3), f"Expected (..., 3, 3) rotation matrix, got {R.shape}"
        assert R_des.shape[-2:] == (3, 3), f"Expected (..., 3, 3) desired rotation matrix, got {R_des.shape}"
        
        R_error = torch.matmul(R_des.transpose(-2, -1), R)
        
        # Extract rotation angle (batched)
        trace = R_error[..., 0, 0] + R_error[..., 1, 1] + R_error[..., 2, 2]
        angle = torch.arccos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
        
        # Extract axis from skew-symmetric part
        small_angle = angle.abs() < 1e-6
        axis = vee_map(R_error - R_error.transpose(-2, -1))
        
        # Avoid division by zero for small angles
        scale = torch.where(small_angle, 
                           torch.ones_like(angle), 
                           angle / (2 * torch.sin(angle)))
        
        return -axis * scale.unsqueeze(-1)
