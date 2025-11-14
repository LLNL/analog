import math
import numpy as np
from aihwkit.linalg import AnalogMatrix

class ABJBlock:
    def __init__(self, rpu, scale, bounds):
        self.rpu = rpu
        self.scale = scale
        self.bounds = bounds

class ABJPreconditioner:
    def __init__(self, matrix, num_blocks, rpu_config):
        self.abj_info = []
        n = matrix.shape[0]
        block_size = math.ceil(n / num_blocks)
        index_bounds = np.arange(0, n, block_size)
        index_bounds = np.append(index_bounds, n)

        for ix in range(num_blocks):
            idx_s, idx_e = index_bounds[ix], index_bounds[ix + 1]
            inv_diag_block = np.linalg.inv(matrix[idx_s:idx_e, idx_s:idx_e])
            scale = np.max(np.abs(inv_diag_block))
            inv_diag_block /= scale
            self.abj_info.append(
                ABJBlock(AnalogMatrix(inv_diag_block.astype("float32"), rpu_config, realistic=False), scale, [idx_s, idx_e])
            )

    def apply(self, in_vector):
        n = in_vector.shape[0]
        in_vector_32 = in_vector.astype("float32")
        out_vector = np.zeros(n, dtype="float64")

        for block in self.abj_info:
            idx_s, idx_e = block.bounds
            out_vector[idx_s:idx_e] = block.scale * (block.rpu.matvec(in_vector_32[idx_s:idx_e])).astype("float64")

        return out_vector