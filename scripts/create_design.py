#!/usr/bin/env python
import os
import sys
from scipy import io
from linescanning import prf
import numpy as np
opj = os.path.join
opd = os.path.dirname

def main(argv):

    """create_design.py

Create full design from log-directory. In another version, we averaged the iterations, which is not done here.

Parameters
----------
    <subject>   subject name
    <log dir>   path to directory containing the .tsv-file and _Screenshots-directory from exptools2
    <output>    output directory
    <n_iters>   optional; number of volumes to use for the design matrix. Defaults to 5200

Returns
----------
    creates <output>/<subject>/<subject>_acq-lines-res-native_desc-full_design.mat

Example
----------
>>> python create_design.py sub-001 sourcedata/sub-001/ses-2/sub-001_ses-2_run-imgs pRFline/data_lp3
    """

    # set defaults
    task    = "task-pRF"
    n_trs   = 5200

    if len(argv) < 3:
        print("NEED MORE ARGUMENTS\n")
        print(main.__doc__)
        sys.exit()

    subject = argv[0]
    log_dir = argv[1]
    out_dir = argv[2]

    if len(argv) > 3:
        n_trs = argv[3]

    dm_f = opj(out_dir, subject, f"{subject}_acq-lines_res-native_desc-full_design.mat")
    if not os.path.exists(os.path.dirname(dm_f)):
        os.makedirs(os.path.dirname(dm_f), exist_ok=True)

    # create full design: baseline + 2 iterations
    if not os.path.exists(dm_f):
        dm = prf.create_line_prf_matrix(
            log_dir, 
            stim_duration=0.25,
            nr_trs=n_trs,
            TR=0.105,
            verbose=True,
            n_pix=100)

        # copy first iteration
        dm_ = dm.copy()
        dm_[...,2476:2476+2286] = dm_[...,190:2476]
        print(f"Writing '{dm_f}'")
        io.savemat(dm_f, {"design": dm_})

if __name__ == "__main__":
    main(sys.argv[1:])
