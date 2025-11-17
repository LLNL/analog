import numpy as np
import scipy.sparse as sparse
import pyamg.krylov as solver
import matplotlib.pyplot as plt

from analog_block_jacobi import ABJPreconditioner

from aihwkit.simulator.presets import ReRamSBPreset
from aihwkit.simulator.presets import IdealizedPreset

def main():
    ## PARAMETER SETUP ##
    # A = Lap + low_rank_coeff*X*X^T + diag_shift*I
    # where Lap is the 2D Laplacian of size n = m^2 and X is of size n x d with d_sparsity
    m = 32
    d = 10
    d_sparsity = 0.10
    low_rank_coeff = 1.0
    diag_shift = 0.0
    num_blocks = [4, 16, 64] # number of blocks to use in analog block Jacobi preconditioning
    tol = 1e-12
    maxit = 200

    # Load IdealizedPreset and match parameters to MATLAB simulator RPU_Analog_Basic.Get_Baseline_Settings
    rpu_config = IdealizedPreset() # idealized preset for testing and comparison to MATLAB code
    rpu_config.mapping.max_input_size = 2048
    rpu_config.mapping.max_output_size = 2048
    rpu_config.device.dw_min_dtod = 0.3 # MATLAB simulator, dw_min_dtod = 0.3
    rpu_config.device.w_max_dtod = 0.2 # MATLAB simulator, weight_range_dtod = 0.2
    rpu_config.device.w_min_dtod = 0.2 # MATLAB simulator, weight_range_dtod = 0.2
    rpu_config.forward.inp_noise = 0.01 # MATLAB simulator, input_noise  = 0.01
    rpu_config.forward.out_noise = 0.02 # MATLAB simulator, output_noise = 0.02
    rpu_config.forward.w_noise = 0.002 # MATLAB simulator, write_noise = 0.002

    ## SET UP PROBLEM AND PRECONDITIONER ##
    # Kronecker sum for 2D Laplacian
    L_onedim = np.diag(2*np.ones(m), 0) + np.diag(-1*np.ones(m-1), 1) + np.diag(-1*np.ones(m-1), -1)
    I_onedim = np.eye(m)
    L = np.kron(L_onedim, I_onedim) + np.kron(I_onedim, L_onedim)
    n = L.shape[0]

    X = sparse.random(n, d, density=d_sparsity)
    X.data = np.random.randn(X.nnz)
    X = X.toarray()
    A = L + low_rank_coeff*np.dot(X, X.T) + diag_shift*np.eye(n)
    D = np.diag(np.diag(A)) # needed for digital Jacobi preconditioning

    b = np.random.normal(size=(n,))
    b = b / np.linalg.norm(b) # random right-hand side vector with norm 1

    # Set up preconditioners and anonymous functions for the SciPy linear operators
    P_info3 = ABJPreconditioner(A, num_blocks[0], rpu_config)
    P_info4 = ABJPreconditioner(A, num_blocks[1], rpu_config)
    P_info5 = ABJPreconditioner(A, num_blocks[2], rpu_config)

    P3 = sparse.linalg.LinearOperator((n, n), matvec=(lambda u: P_info3.apply(u)))
    P4 = sparse.linalg.LinearOperator((n, n), matvec=(lambda u: P_info4.apply(u)))
    P5 = sparse.linalg.LinearOperator((n, n), matvec=(lambda u: P_info5.apply(u)))

    ## RUN GMRES AND FGMRES ##

    resvec1, resvec2, resvec3, resvec4, resvec5 = [], [], [], [], []

    x1, flag1 =  solver.gmres(A, b, x0=np.zeros(n), tol=tol, restart=None, maxiter=maxit, M=None, residuals=resvec1)
    x2, flag2 =  solver.gmres(A, b, x0=np.zeros(n), tol=tol, restart=None, maxiter=maxit, M=np.linalg.inv(D), residuals=resvec2)
    x3, flag3 = solver.fgmres(A, b, x0=np.zeros(n), tol=tol, restart=None, maxiter=maxit, M=P3, residuals=resvec3)
    x4, flag4 = solver.fgmres(A, b, x0=np.zeros(n), tol=tol, restart=None, maxiter=maxit, M=P4, residuals=resvec4)
    x5, flag5 = solver.fgmres(A, b, x0=np.zeros(n), tol=tol, restart=None, maxiter=maxit, M=P5, residuals=resvec5)

    ## VISUALIZATION ##

    x_true = np.linalg.solve(A, b)
    print("Absolute errors:", np.linalg.norm(x_true - x1), np.linalg.norm(x_true - x2), np.linalg.norm(x_true - x3), np.linalg.norm(x_true - x4), np.linalg.norm(x_true - x5))

    plt.semilogy(resvec1, '-k', linewidth=1.2, label="GMRES")
    plt.semilogy(resvec2, '--k', linewidth=1.2, label="GMRES (Jacobi)")
    plt.semilogy(resvec3, '-r', linewidth=1.2, label="FGMRES (ABJ%i)" % (num_blocks[0]))
    plt.semilogy(resvec4, '-g', linewidth=1.2, label="FGMRES (ABJ%i)" % (num_blocks[1]))
    plt.semilogy(resvec5, '-b', linewidth=1.2, label="FGMRES (ABJ%i)" % (num_blocks[2]))
    plt.xlabel("Iteration number")
    plt.ylabel("Relative residual norm")
    plt.title("$A = -\\nabla^2 + XX^T + \\alpha I$, n = %i, d = %i, $\\alpha$ = %.2f" % (n, d, diag_shift))
    plt.legend(loc='lower left')
    plt.savefig("fgmres_abj_comparison.pdf", format="pdf")

if __name__ == '__main__':
    main()