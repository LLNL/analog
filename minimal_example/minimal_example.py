import numpy as np
from aihwkit.linalg import AnalogMatrix
from aihwkit.simulator.presets import ReRamSBPreset

def main():
    # aihwkit defaults to single precision; they mention there is a C++ compiler option to
    # use double precision, but I cannot find how to do this or have it work with pip installation;
    # possibly something we should ask Tayfun
    A = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]).astype("float32")
    x = np.array([[0.7, 0.7], [-0.4, -0.4], [0.2, 0.2]]).astype("float32")

    exact_mult = A @ x[:, 0]

    rpu_config = ReRamSBPreset() # preset configuration option
    P = AnalogMatrix(A, rpu_config, realistic=False)

    analog_mult1 = P.matvec(x[:, 0])
    analog_mult2 = P.matvec(x[:, 1])

    # matmat complains about a dimensionality issue that I do not fully understand yet
    # since we are simply performing (3x3) x (3x2) --> (3x2)
    # analog_matmat = P.matmat(x)

    print("Exact:  ", exact_mult)
    print("Analog: ", analog_mult1)
    print("Analog: ", analog_mult2)

if __name__ == "__main__":
    main()
