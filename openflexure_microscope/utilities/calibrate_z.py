from __future__ import print_function
from .. import microscope
import picamera.array
import picamera
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

def prep_z_compensation_table(datafile, settings,min_score=1000):
    """Load a data file formatted as x y z score and save it
    """
    with microscope.load_microscope(settings) as ms:
        settings = ms.settings_dict()
        data=np.loadtxt(datafile)
        #remove bad datapoints
        data=data[data[:,3]>1000]
        data=data[:,0:3]
        settings['z_compensation_table']=np.swapaxes(data,0,1)

    for k in settings:
        print("{}: {}".format(k, settings[k]))
    np.savez(output_fname, **settings)
    print("Z calibration table saved to {}".format(output_fname))

if __name__ == '__main__':
        generate_lens_shading_table_closed_loop(sys.argv[1],sys.argv[2])
