from __future__ import print_function
import numpy as np
import sys
import time

method="sphere"
#method=interpolate

def prep_z_compensation_table(datafile, settingsfile,min_score=1000):
    """Load a data file formatted as x y z score and save it
    """
    settings = dict(np.load(settingsfile))
    data=np.loadtxt(datafile)
    #remove bad datapoints
    data=data[data[:,3]>1000]
    data=data[:,0:3]
    settings['z_compensation_table']=np.swapaxes(data,0,1)
    settings['z_compensation_method']="interpolate"

    for k in settings:
        print("{}: {}".format(k, settings[k]))
    np.savez(settingsfile, **settings)
    print("Z calibration table saved to {}".format(settingsfile))

def prep_sphere(settings_file):
    maxs=int(input("Grid size:"))
    points=int(input("Grid points in line:"))
    print("Calibrating on {} points.".format(points*points))
    spacing=maxs//points
    totalpoints=points*points
    done=0
    with microscope.load_microscope(settings_file) as ms:
        data=[]
        xdir=1
        camera = ms.camera
        stage = ms.stage
        stage.backlash=256
        ms.stage.z_compensation_table = None
        dz_fine=np.linspace(-60,60,13)
        dz_rough=np.linspace(-250,250,11)
        dz_hyperrough=np.linspace(-1500,1500,15)
        for i in range(0, points):
            for j in range(0, points):
                print("Done: {}".format(round(i*points+j)/total*100.0))
                cp=ms.stage.position
                print("processing: {}".format(cp))
                ms.autofocus(dz_hyperrough)
                ms.autofocus(dz_rough)
                ms.autofocus(dz_fine)
                data.append(ms.stage.position)
                if not j == points-1:
                    ms.stage.move_rel([spacing*xdir,0])
            ms.stage.move_rel([0,spacing,0])
            xdir*=-1
        print(data)
        data=np.array(data)
        xy=np.swapaxes(data[:,0:2],0,1)
        z=data[:,2]
        #TODO: fix r?
        params,covars=scipy.optimize.curve_fit(f,xy,z,p0=[500000,cp[0],cp[1],cp[2]])
        print(params)
        print(covars)
        ms.stage.z_compensation_method="sphere"
        ms.stage.z_compensation_params=params
        ms.save_settings(settings_file)

if __name__ == '__main__':
    if method == "interpolate":
        prep_z_compensation_table(sys.argv[1],sys.argv[2])
    else if method="sphere":
        prep_sphere(sys.argv[1])
