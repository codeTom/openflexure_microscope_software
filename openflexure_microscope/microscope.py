from __future__ import print_function
import picamera
from picamera import PiCamera
import picamera.array
import numpy as np
import scipy
from scipy import ndimage
import time
import matplotlib.pyplot as plt
from openflexure_stage import OpenFlexureStage
from contextlib import contextmanager, closing
import os

picam2_full_res = (3280, 2464)
picam2_half_res = tuple([d/2 for d in picam2_full_res])
picam2_quarter_res = tuple([d/4 for d in picam2_full_res])
# The dicts below determine the settings that are loaded from, and saved to,
# the npz settings file.  See extract_settings and load_microscope for details.
picamera_init_settings = {"lens_shading_table": None, "resolution": tuple}
picamera_later_settings = {"awb_mode":str,
                           "awb_gains":tuple,
                           "shutter_speed":"[()]",
                           "analog_gain":"[()]",
                           "digital_gain":"[()]",
                           "brightness":"[()]",
                           "contrast":"[()]",
                           }

def picamera_supports_lens_shading():
    """Determine whether the picamera module supports lens shading.

    As of March 2018, picamera did not wrap the necessary MMAL commands to
    set the lens shading table, or to write the value of analog or digital
    gain.  I have a forked version of the library that does support these.

    For ease of use by people who don't want those features, this library
    does not have a hard dependency on lens shading.  However, we need to
    check in some places whether it's available.
    """
    return hasattr(PiCamera, "lens_shading_table")

# For backwards compatibility, remove settings that aren't available
if not picamera_supports_lens_shading():
    # remove settings from the dictionaries that aren't available
    del picamera_init_settings['lens_shading_table']
    del picamera_later_settings['analog_gain']
    del picamera_later_settings['digital_gain']
    print("WARNING: the currently-installed picamera library does not support all "
          "the features of the openflexure microscope software.  These features "
          "include lens shading control and setting the analog/digital gain.\n"
          "\n"
          "See the installation instructions for how to fix this:\n"
          "https://github.com/rwb27/openflexure_microscope_software")

def round_resolution(res):
    """Round up the camera resolution to units of 32 and 16 in x and y"""
    return tuple([int(q*np.ceil(res[i]/float(q))) for i, q in enumerate([32,16])])

def decimate_to(shape, image):
    """Decimate an image to reduce its size if it's too big."""
    decimation = np.max(np.ceil(np.array(image.shape, dtype=np.float)[:len(shape)]/np.array(shape)))
    return image[::int(decimation), ::int(decimation), ...]

def sharpness_sum_lap2(rgb_image):
    """Return an image sharpness metric: sum(laplacian(image)**")"""
    image_bw=np.mean(decimate_to((1000,1000), rgb_image),2)
    image_lap=ndimage.filters.laplace(image_bw)
    return np.mean(image_lap.astype(np.float)**4)

def sharpness_edge(image):
    """Return a sharpness metric optimised for vertical lines"""
    gray = np.mean(image.astype(float), 2)
    n = 20
    edge = np.array([[-1]*n + [1]*n])
    return np.sum([np.sum(ndimage.filters.convolve(gray,W)**2)
                   for W in [edge, edge.T]])
@contextmanager
def set_properties(obj, **kwargs):
    """A context manager to set, then reset, certain properties of an object.

    The first argument is the object, subsequent keyword arguments are properties
    of said object, which are set initially, then reset to their previous values.
    """
    saved_properties = {}
    for k in kwargs.keys():
        try:
            saved_properties[k] = getattr(obj, k)
        except AttributeError:
            print("Warning: could not get {} on {}.  This property will not be restored!".format(k, obj))
    for k, v in kwargs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in saved_properties.items():
            setattr(obj, k, v)


class Microscope(object):
    def __init__(self, camera=None, stage=None):
        """Create the microscope object.  The camera and stage should already be initialised."""
        self.camera = camera
        self.stage = stage
        self.stage.backlash = np.zeros(3, dtype=np.int)

    def close(self):
        """Shut down the microscope hardware."""
        self.camera.close()
        self.stage.close()

    def rgb_image_old(self, use_video_port=True):
        """Capture a frame from a camera and output to a numpy array"""
        res = round_resolution(self.camera.resolution)
        shape = (res[1], res[0], 3)
        buf = np.empty(np.product(shape), dtype=np.uint8)
        self.camera.capture(buf,
                format='rgb',
                use_video_port=use_video_port)
        #get an image, see picamera.readthedocs.org/en/latest/recipes2.html
        return buf.reshape(shape)

    def rgb_image(self, use_video_port=True, resize=None):
        """Capture a frame from a camera and output to a numpy array"""
        with picamera.array.PiRGBArray(self.camera, size=resize) as output:
#            output.truncate(0)
            self.camera.capture(output,
                    format='rgb',
                    resize=resize,
                    use_video_port=use_video_port)
        #get an image, see picamera.readthedocs.org/en/latest/recipes2.html
            return output.array

    def freeze_camera_settings(self, iso=None, wait_before=2, wait_after=0.5):
        """Turn off as much auto stuff as possible (except lens shading)

        NB if the camera was created with load_microscope, this is not necessary.
        """
        if iso is not None:
            self.camera.iso = iso
        time.sleep(wait_before)
        self.camera.shutter_speed = self.camera.exposure_speed
        self.camera.exposure_mode = "off"
        g = self.camera.awb_gains
        self.camera.awb_mode = "off"
        self.camera.awb_gains = g
        print("Camera settings are frozen.  Analogue gain: {}, Digital gain: {}, Exposure speed: {}, AWB gains: {}".format(self.camera.analog_gain, self.camera.digital_gain, self.camera.exposure_speed, self.camera.awb_gains))
        time.sleep(wait_after)

    def autofocus(self, dz, settle=0.5, metric_fn=sharpness_sum_lap2):
        """Perform a simple autofocus routine.

        The stage is moved to z positions (relative to current position) in dz,
        and at each position an image is captured and the sharpness function
        evaulated.  We then move back to the position where the sharpness was
        highest.  No interpolation is performed.

        dz is assumed to be in ascending order (starting at -ve values)
        """
        with set_properties(self.stage, backlash=256):
            sharpnesses = []
            positions = []
            self.camera.annotate_text = ""
            for i in self.stage.scan_z(dz, return_to_start=False):
                positions.append(self.stage.position[2])
                time.sleep(settle)
                sharpnesses.append(metric_fn(self.rgb_image(
                            use_video_port=True,
                            resize=(640,480))))
            newposition = positions[np.argmax(sharpnesses)]
            self.stage.focus_rel(newposition - self.stage.position[2])
            return positions, sharpnesses


    def naive_autofocus(self, step_size = 500, thresh = 999999):
        """
        IGEM15
        """
        print("Is simple better?")
        prev = self.get_score()

        self.stage.focus_rel(step_size)
        time.sleep(0.4)
        curr = self.get_score()
        print(prev)
        print(curr)

        if curr > thresh:
            print("tres focused")
            return

        # check direction to climb
        diff = curr - prev
        direction = 1 if diff > 0 else -1

        # climb in smaller increments until theshold
        step_thresh = 5

        # for sanity
        timeout = 100
        tprogress = 0

        # climb, baby, climb!
        while step_size > step_thresh:
            while tprogress < timeout:
                prev = curr
                self.stage.focus_rel(direction * step_size)
                time.sleep(0.4)
                curr = self.get_score()
                print(curr)
                if curr > thresh:
                    print("tres focused")
                    return

                diff = curr - prev
                curdir = 1 if diff > 0 else -1
                direction = curdir * direction
                if curdir == -1:
                    print("back back back")
                    self.stage.focus_rel(direction * step_size)
                    if thresh == 999999:
                        thresh = prev#naive_autofocus(f, step_size, prev - 100)
                        #return
                    break
                tprogress = tprogress + 1
            tprogress = 0
            step_size = step_size / 1.5

        print("Done naive hill climb")
        print("Final score {}".format(curr))

    def autofocus_npoint_fine(self, n, step):
        """
            Get n points around current positons, with step between them
            Use a parabolic fit to estimate the optimal position
            assumes the microscope is in a stable state (takes the first image immediately)
            return: the relative position, which can then be used to extrapolate approx next location
        """
        with set_properties(self.stage, backlash=256):
            sharpnesses = []
            positions = []
            self.camera.annotate_text = ""
            settle=0.5
            #check if we scan current position
            dz=np.linspace(-(n//2)*step,(n//2)*step,n)
            scanstart=time.time()
            if n%2 == 1:
                positions.append(0)
                sharpnesses.append(self.get_score())
                dz=np.delete(dz,n//2,0)
#                print(dz)
            for i in self.stage.scan_z(dz, return_to_start=False):
                time.sleep(settle)
                sharpnesses.append(self.get_score())
            scanend=time.time()
            print("scan took: {}".format(scanend-scanstart))
            # Fit a parabola over these points
#            print(bestpos)
#            print(dz)
            dz=np.insert(dz,0,0,0)
#           print(sharpnesses)
            coeffs=np.polyfit(dz,sharpnesses,2)
            bestpos=-coeffs[1]/(2*coeffs[0])
#            print(coeffs)
            if(coeffs[0]>0):
                print("Something went horribly wrong, try again")
                if sharpnesses[1]>sharpnesses[n-1]:
                    self.stage.focus_rel(-2*(n//2)*step) #go to the better endpoint
                return False

            print("fitting took:{}".format(time.time()-scanend))
            self.stage.focus_rel(bestpos-dz[n-1])
#           newposition = positions[np.argmax(sharpnesses)]
#           self.stage.focus_rel(newposition - self.stage.position[2])
#           return positions, sharpnesses
        return bestpos



    def autofocus_igem(self, step_size):
        """
            Autofocus copied from iGEM15 Cambridge team,
            https://github.com/sourtin/igem15-sw

            step_size: initial step size
        """
        z,f = self.hill_climbing(step_size)
        #use parabola prediction
        zb = self.parabola_fitting(z,f)
        print("hill climb returned:\n")
        print(z)
        print(f)
        self.stage.focus_rel(z[2]-zb)
        print("Parabola fit:\n")
        print(zb)
        print("score: {}".format(self.get_score()))
        #TODO


    def parabola_fitting(self, z , f):
        """Fit the autofocus function data according to the equation 16.5 in the 
           textbook : 'Microscope Image Processing' by Q.Wu et al'
           parabola_fitting((z1, z2, z3), (f1, f2 f3))
           where z1, z2 , z3 are points where the autofocus functions 
           values f1, f2, f3 are measured
           Taken from iGEM15
        """
        z1 = z[0];z2 = z[1];z3 = z[2];    
        f1 = f[0];f2 = f[1];f3 = f[2];
        E = (f2 - f1)/(f3 - f2)
        if (z3 - z2) == (z2 - z1):
            return 0.5 * (E * (z3 + z2) - (z2 + z1))/(E - 1)
        else:
            return 0.5 * (E * (z3 ** 2 - z2 ** 2) - (z2 ** 2 - z1 ** 2))/(E * (z3 - z2) - (z2 - z1))

    def hill_climbing(self, step_size):
        """
            Autofocus copied from iGEM15 Cambridge team,
            https://github.com/sourtin/igem15-sw

            step_size: initial step size
        """
        score_history = []

        with set_properties(self.stage, backlash=256):
            """ Climb to a higher place, find a smaller interval containing focus position
            (z1, z2, z3),(f1, f2, f3) = hill_climbing(f)
            """
            global scan_direction
            print('Starting hill climbing')
            f1 = self.get_score()
            z1 = 0
            self.stage.focus_rel(step_size)
            time.sleep(0.4)
            f2 = self.get_score()
            z2 = step_size
            score_history.append(f1)
            score_history.append(f2)
            iterations = 0

            while(1):
                #print(f1,f2)
                if(f2 > f1):
                    f0 = f1
                    f1 = f2
                    self.stage.focus_rel(step_size)
                    time.sleep(0.4)
                    f2 = self.get_score()
                    score_history.append(f2)
                    z2 += step_size
                    iterations += 1

                elif(iterations <=1):
                    #print(iterations, f1, f2)
                    print('Found a dip, assuming it is wrong and continuing')
                    f0 = f1
                    f1 = f2
                    self.stage.focus_rel(step_size)
                    time.sleep(0.4)
                    f2 = self.get_score()
                    score_history.append(f2)
                    z2 += step_size
                    iterations +=1
                elif(iterations <= 2):
                    print('Changing search direction')
                    return self.hill_climbing(-step_size)
                else:
                    print ('Finished hill climbing')
                    return ((z2 - 2 * step_size -2 , z2 - step_size, z2), (f0, f1, f2))

    def get_score(self):
        return sharpness_sum_lap2(self.rgb_image(
                            use_video_port=True,
                            resize=(640,480)))
#        return np.var(self.rgb_image(
#                            use_video_port=True,
#                            resize=(640,480)))

    def test_score_repeatability(self):
        """
            Tests the stability of the focus score to unavoidable vibrations
        """
        sleep_t=0.1
        scores=[]
        for i in range(50):
            score=self.get_score()
            print("Iteration {}: {}".format(i,score))
            scores.append(score)
        return scores

    def test_z_axis_repeatability(self):
        """
            Testing only. TODO: remove (also from iGEM15)
        """
        scores = []
        distance = 800
        for i in range(50):
            print('Iteration {}'.format(i))
            self.stage.focus_rel(distance)
            self.stage.focus_rel(-distance)
            #scores.append(m.move_motor(-distance, 5).eval_score())
            scores.append(self.get_score())


        print(scores)
        plt.plot(range(len(scores)),scores)
        plt.ylabel('Variance')
        plt.xlabel('Iteration')
        plt.title('Focus score varying repeated movements')
        plt.show()

    def acquire_image_stack(self, step_displacement, n_steps, output_dir, raw=False):
        """Scan an edge across the field of view, to measure distortion.

        You should start this routine with the edge positioned in the centre of the
        microscope's field of view.  You specify the x,y,z shift between images, and
        the number of images - these points will be distributed either side of where
        you start.

        step_displacement: a 3-element array/list specifying the step (if a scalar
            is passed, it's assumed to be Z)
        n_steps: the number of steps to take
        output_dir: the directory in which to save images
        backlash: the backlash correction amount (default: 128 steps)
        """
        # Ensure the displacement per step is an array, and that scalars do z steps
        step_displacement = np.array(step_displacement)
        if len(step_displacement.shape) == 0:
            step_displacement = np.array([0, 0, step_displacement.value])
        elif step_displacement.shape == (1,):
            step_displacement = np.array([0, 0, step_displacement[0]])
        ii = np.arange(n_steps) - (n_steps - 1.0)/2.0 # an array centred on zero
        scan_points = ii[:, np.newaxis] * step_displacement[np.newaxis, :]

        with set_properties(self.stage, backlash=256):
            for i in self.stage.scan_linear(scan_points):
                time.sleep(1)
                filepath = os.path.join(output_dir,"image_%03d_x%d_y%d_z%d.jpg" %
                                                   ((i,) + tuple(self.stage.position)))
                print("capturing {}".format(filepath))
                self.camera.capture(filepath, use_video_port=False, bayer=raw)
            time.sleep(0.5)

    def settings_dict(self):
        """Return all the relevant settings as a dictionary."""
        settings = {}
        for k in list(picamera_later_settings.keys()) + list(picamera_init_settings.keys()):
            settings[k] = getattr(self.camera, k)
        return settings

    def save_settings(self, npzfile):
        "Save the microscope's current settings to an npz file"
        np.savez(npzfile, **self.settings_dict())

    @property
    def zoom(self):
        """A scalar property that sets the zoom value of the camera.

        camera.zoom is a 4-element field of view specifying the region of the
        sensor that is visible, this is a simple scalar, where 1 means the whole
        FoV is returned and >1 means we zoom in (on the current centre of the
        FoV, which may or may not be (0.5,0.5)
        """
        fov = self.camera.zoom
        return 2.0/(fov[2] + fov[3])

    @zoom.setter
    def zoom(self, newvalue):
        """Set the zoom of the camera, keeping the current image centre"""
        if newvalue < 1.0:
            newvalue = 1.0
        fov = self.camera.zoom
        centre = np.array([fov[0] + fov[2]/2.0, fov[1] + fov[3]/2.0])
        size = 1.0/newvalue
        # If the new zoom value would be invalid, move the centre to
        # keep it within the camera's sensor (this is only relevant
        # when zooming out, if the FoV is not centred on (0.5, 0.5)
        for i in range(2):
            if np.abs(centre[i] - 0.5) + size/2 > 0.5:
                centre[i] = 0.5 + (1.0 - size)/2 * np.sign(centre[i]-0.5)
        print("setting zoom, centre {}, size {}".format(centre, size))
        new_fov = (centre[0] - size/2, centre[1] - size/2, size, size)
        self.camera.zoom = new_fov


def extract_settings(source_dict, converters):
    """Extract a subset of a dictionary of settings.

    For each item in ``source_dict`` that shares a key with an item in
    ``converters``, return a dictionary of values that have been
    processed using the conversion functions in the second dict.

    NB "None" is equivalent to no processing, to save some typing.
    There are some special string values for converters:
    "[()]" will convert a 0-dimensional numpy array to a scalar
    "[0]" will return the first element of a 1D array
    If either "[()]" or "[0]" is specified and raises an exception,
    then we fall back to no processing.  This is good if the values
    might be from a numpy ``.npz`` file, or might be specified directly.
    """
    settings = {}
    for k in source_dict:
        if k in converters:
            if converters[k] is None:
                settings[k] = source_dict[k]
            elif converters[k] == "[()]":
                try:
                    settings[k] = source_dict[k][()]
                except:
                    settings[k] = source_dict[k]
            elif converters[k] == "[0]":
                try:
                    settings[k] = source_dict[k][0]
                except:
                    settings[k] = settings[k]
            else:
                settings[k] = converters[k](source_dict[k])
    return settings


class DummyStage():
    position = np.array([0,0,0])
    def move(self, *args, **kwargs):
        pass
    def move_rel(self, *args, **kwargs):
        pass
    def close(self):
        pass

@contextmanager
def load_microscope(npzfile=None, save_settings=False, dummy_stage=True, **kwargs):
    """Create a microscope object with specified settings. (context manager)

    This will read microscope settings from a .npz file, and/or from
    keyword arguments.  It will then create the microscope object, and
    close it at the end of the with statement.  Keyword arguments will
    override settings specified in the file.

    If save_settings is
    True, it will attempt to save the microscope's settings at the end of
    the with block, to the same filename.  If save_settings_on_exit is set
    to a string, it should save instead to that filename.
    """
    settings = {}
    try:
        npz = np.load(npzfile)
        for k in npz:
            settings[k] = npz[k]
    except:
        pass
    settings.update(kwargs)

    if "stage_port" in settings:
        stage_port = settings["stage_port"]
        del settings["stage_port"]
    else:
        stage_port = None

    # Open the hardware connections
    with closing(DummyStage() if dummy_stage else OpenFlexureStage(stage_port)) as stage, \
         closing(PiCamera(**extract_settings(settings, picamera_init_settings))) as camera:
        ms = Microscope(camera, stage)
        for k, v in extract_settings(settings, picamera_later_settings).items():
            setattr(ms.camera, k, v)
        yield ms # The contents of the with block from which we're called happen here
        if save_settings:
            if save_settings is True:
                save_settings = npzfile
            ms.save_settings(save_settings)


if __name__ == "__main__":
    with picamera.PiCamera() as camera, \
         OpenFlexureStage("/dev/ttyUSB0") as stage:
#        camera.resolution=(640,480)
        camera.start_preview()
        ms = Microscope(camera, stage)
        ms.freeze_camera_settings(iso=100)
        camera.shutter_speed = camera.shutter_speed / 4

        backlash=128

        for step,n in [(1000,10),(200,10),(100,10),(50,10)]:
            dz = (np.arange(n) - (n-1)/2.0) * step

            pos, sharps = ms.autofocus(dz, backlash=backlash)


            plt.plot(pos,sharps,'o-')

        plt.xlabel('position (Microsteps)')
        plt.ylabel('Sharpness (a.u.)')
        time.sleep(2)

    plt.show()

    print("Done :)")

