import numpy as np
from importlib.resources import as_file, files

RESOURCES = files("nDGPemu")

def test_predict(model):
    # set reference parameters
    cosmo_params = {'Om':0.3089,
                'ns':0.9667,
                'As':2.066e-9,
                'h':0.6774,
                'Ob':0.0486}
    H0rc = 1
    z = 1

    Bk = model.predict(H0rc,z,cosmo_params)
    with as_file(RESOURCES / "cache/Test_Bk.npy") as myfile:
        Bk_ref = np.load(myfile, allow_pickle=True)

    assert all(abs(Bk-Bk_ref)<1e-7) , f"Test failed: the model could not reproduce the reference boost factor."

    print("All tests passed successfully.")
