import numpy as np
import scipy.sparse as sp
from functools import reduce

def sparse_first_derivative(V):
    if V < 2:
        raise ValueError("At least 2 continue voxels to calculate 1st order derivative.")
    
    diags = []
    offsets = []
    # main diagonal zeros except boundaries
    main = np.zeros(V)
    diags.append(main); offsets.append(0)

    # sub and super
    sub = np.zeros(V-1)
    sup = np.zeros(V-1)
    sub[:] = -0.5
    sup[:] = +0.5
    diags.append(sub); offsets.append(-1)
    diags.append(sup); offsets.append(1)
    D = sp.diags(diags, offsets, shape=(V, V), format='csr')

    D = D.tolil()
    D[0, :] = 0
    D[0, 0] = -1
    D[0, 1] = 1
    D[-1, :] = 0
    D[-1, -2] = -1
    D[-1, -1] = 1
    return D.tocsr()

def sparse_second_derivative(V):
    if V < 3:
        raise ValueError("At least 3 continue voxels to calculate 2nd order derivative.")

    D = sp.diags(
        [np.ones(V-1), -2*np.ones(V), np.ones(V-1)],
        offsets=[-1, 0, 1],
        shape=(V, V),
        format='csr'
    ).tolil()

    D[0, :] = 0
    # D[0, 0] = 1
    # D[0, 1] = -2
    # D[0, 2] = 1
    D[-1, :] = 0
    # D[-1, -3] = 1
    # D[-1, -2] = -2
    # D[-1, -1] = 1
    return D.tocsr()

def R_all(v, m, order=1):
    """
    v:  an int or a tuple. Like 2, or (2, 3), or (5, 6, 7). Length is up to 3
    m:  the length of q-vector
    """
    if not isinstance(v, (int, tuple)):
        raise ValueError("v must be an int or a tuple of ints.")

    if isinstance(v, int):
        v = (v,)

    if len(v) > 3:
        raise ValueError("v only supports up to 3D!")

    n = len(v)

    funs = {1: sparse_first_derivative, 2: sparse_second_derivative}

    D_list = [funs[order](element) for element in v]                    # it stores D1, D2, D3
    I_list = [sp.eye(element, format='csr') for element in v]           # it stores I1, I2, I3

    def kron_seq(mats):
        return reduce(lambda a, b: sp.kron(a, b, format='csr'), mats)   # it calculates kron(kron(mats[0], mats[1]), mats[2])

    terms = []
    for i in range(n):
        kron_mats = []
        for j in range(n):
            kron_mats.append(D_list[j] if j == i else I_list[j])        # kron_mats stores [D1, I2, I3], etc
        terms.append(kron_seq(kron_mats))
    D_base = sum(terms)

    R = sp.kron(D_base.T.dot(D_base), sp.eye(m, format='csr'), format='csr')
    return R

def R_mask(mask, m, order=1):
    """
    mask:   to deal with real data, we need to input a 3D mask array to tell which region is brain in a cube
    m:      the length of q-vector
    """

    if isinstance(mask, np.ndarray) and mask.dtype == bool and mask.ndim == 3:
        coords = np.argwhere(mask)

    N = coords.shape[0]

    # generate the smallest cube containing the mask
    minc = coords.min(axis=0)
    maxc = coords.max(axis=0)
    shape = tuple((maxc - minc) + 1)
    coords = coords - minc                  # calculate the coords in the croped cube

    # lin2idx: a 3D array storing the order of mask==True voxels in the coordinate system of the smallest cube
    # if mask == False, then -1
    lin2idx = -np.ones(shape, dtype=int)
    lin_inds = np.ravel_multi_index(coords.T, dims=shape, order='C')
    lin2idx_flat = lin2idx.ravel()
    lin2idx_flat[lin_inds] = np.arange(N)
    lin2idx = lin2idx_flat.reshape(shape)

    # coo sparse information helper to gather neighbor relationships axis-wise
    rows = []
    cols = []
    data = []

    # offsets along axes: axis 0 -> x, axis 1 -> y, axis 2 -> z in coords
    for axis in range(3):
        # roll arrays to get backward and forward neighbor indices
        idx_backwork = np.roll(lin2idx, shift=+1, axis=axis)
        idx_forward  = np.roll(lin2idx, shift=-1, axis=axis)

        # compute mask of voxels having physically valid forward and backward voxels
        coords_axes = np.indices(shape)[axis]
        mask_backward = coords_axes > 0
        mask_forward  = coords_axes < (shape[axis] - 1)

        mask_central = (idx_backwork != -1) & (lin2idx != -1) & (idx_forward != -1) & mask_backward & mask_forward

        if order == 1:
            if np.any(mask_central):
                ci = lin2idx[mask_central]
                ib = idx_backwork[mask_central]
                ip = idx_forward[mask_central]

                rows.append(np.repeat(ci, 3))
                cols.append(np.column_stack([ib, ci, ip]).ravel())
                data.append(np.tile([-0.5, 0.0, 0.5], len(ci)))

            mask_bwd = (idx_backwork != -1) & (lin2idx != -1) & (~mask_central) & mask_backward
            if np.any(mask_bwd):
                ib = idx_backwork[mask_bwd]
                ci = lin2idx[mask_bwd]
                rows.append(np.repeat(ci, 2))
                cols.append(np.column_stack([ib, ci]).ravel())
                data.append(np.tile([-1.0, +1.0], len(ci)))

            mask_fwd = (idx_forward != -1) & (lin2idx != -1) & (~mask_central) & mask_forward
            if np.any(mask_fwd):
                ip = idx_forward[mask_fwd]
                ci = lin2idx[mask_fwd]
                rows.append(np.repeat(ci, 2))
                cols.append(np.column_stack([ci, ip]).ravel())
                data.append(np.tile([-1.0, +1.0], len(ci)))

        elif order == 2:
            if np.any(mask_central):
                ci = lin2idx[mask_central]
                ib = idx_backwork[mask_central]
                ip = idx_forward[mask_central]
                rows.append(np.repeat(ci, 3))
                cols.append(np.column_stack([ib, ci, ip]).ravel())
                data.append(np.tile([1.0, -2.0, 1.0], len(ci)))
        else:
            raise ValueError("order must be 1 or 2")

    if len(rows) == 0:
        D = sp.csr_matrix((N, N))
    else:
        row = np.concatenate(rows).astype(int)
        col = np.concatenate(cols).astype(int)
        data = np.concatenate(data).astype(float)
        D = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()

    R = sp.kron(D.T.dot(D), sp.eye(m, format='csr'), format='csr')

    return R

