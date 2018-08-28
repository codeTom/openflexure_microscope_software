from __future__ import print_function
import numpy as np
import sys
import time

def prep_z_compensation_table(datafile, settingsfile,min_score=1000):
    """Load a data file formatted as x y z score and save it
    """
    settings = dict(np.load(settingsfile))
    data=np.loadtxt(datafile)
    #remove bad datapoints
    data=data[data[:,3]>1000]
    data=data[:,0:3]
    settings['z_compensation_table']=np.swapaxes(data,0,1)

    for k in settings:
        print("{}: {}".format(k, settings[k]))
    np.savez(settingsfile, **settings)
    print("Z calibration table saved to {}".format(settingsfile))

if __name__ == '__main__':
        prep_z_compensation_table(sys.argv[1],sys.argv[2])
