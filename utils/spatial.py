import numpy as np
from scipy.spatial.transform import Rotation

def homogenious_matrix(R=np.eye(3), t=np.zeros(3)):
    R, t = np.array(R), np.array(t).squeeze()
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,-1] = t
    return T

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def normalize_rotation_matrix(R):
    # note: it is easier to normalize the quaternions than the rotation matrix,
    #       therefore transform into quaternions first
    q = Rotation.from_matrix(R).as_quat()
    q_normalized = normalize_quaternion(q)
    R_normalized = Rotation.from_quat(q_normalized).as_matrix()
    return R_normalized

def kabsch_umeyama(A, B):
    '''Kabsch-Umeyama algorithm is a method for aligning and comparing the similarity
    between two sets of points. It finds the optimal translation, rotation and scaling
    by minimizing the root-mean-square deviation (RMSD) of the point pairs.
    
    Sources:
    - Kabsch, W. (1976). "A solution for the best rotation to relate two sets of vectors".
    Acta Crystallographica. A32 (5): 922-923. doi:10.1107/S0567739476001873.
    - Kabsch, W. (1978). "A discussion of the solution for the best rotation to relate two
    sets of vectors". Acta Crystallographica. A34 (5): 827-828. doi:10.1107/S0567739478001680
    - Umeyama, S. (1991). "Least-squares estimation of transformation parameters between two
    point patterns". IEEE Transactions on Pattern Analysis and Machine Intelligence.
    doi:10.1109/34.88573.
    
    Code was taken from:
    https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
    '''
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t