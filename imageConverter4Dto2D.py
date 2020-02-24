import numpy as np
import nibabel as nib
import glob
import os 
import cv2
import time
import argparse
############
"""
This script reads nii.gz fMRI file which are 4-D data and extract slices and store them in the disk. This process refers
to fMRI 4D decomposition to 2D PNG image / slices using a lossless method.
To run this image converter, it is required to install some Python packages listed as follows:
opencv : computer vision library 
nibabel : medical imaging library
This script can be used for any research purposed by citing the following manuscript:
MCADNNet: Recognizing Stages of Cognitive Impairment Through Efficient Convolutional fMRI and MRI Neural Network Topology Models
https://ieeexplore.ieee.org/document/8883215
For any questions or feedback please email samansarraf@ieee.org
Command example:
python imageConverter_persubject_forXin.py --inputfolder Users\image --targetfolder Users\image\out --prefix AD --pattern PreProc --drop 5 --verbose True
Shorter command:
python imageConverter_persubject_forXin.py --inputfolder Users\image --targetfolder Users\image\out --prefix AD
"""


def imgconverting(foldername , targetfolder, prefix, pattern, slicedrop = 10, printfilename=True):
    start = time.time()
    folder = foldername
    os.chdir(folder)
    listfiles = glob.glob('*.nii.gz')
    listshort = listfiles 
    pref = prefix
    if not os.path.isdir(targetfolder):
        os.mkdir(targetfolder)
    if not os.path.isdir(targetfolder):
        os.mkdir(targetfolder)
    for hh in range(len(listshort)):
        img_test = nib.load(listshort[hh]).get_data()
        print(np.shape(img_test))
        slice3 = np.shape(img_test)[2]
        slice4 = np.shape(img_test)[3]
        if img_test.sum() != 0:
            img_test = (img_test/np.amax(img_test))*255
        for ii in range(slice3-slicedrop):
            for jj in range(slice4):
                img_png = img_test[:,:,ii,jj]
                img_png = np.float64(img_png)
                img_png = img_png.astype(np.uint8)
                if not pattern:
                    helper = ('%s_%s_%03d_%03d_%03d.png' % (
                    pref, listshort[hh].replace('.nii.gz', ''), hh + 1, ii + 1, jj + 1))
                    filename = os.path.join(targetfolder, helper)
                else:
                    helper = ('%s_%s_%03d_%03d_%03d.png' % (pref, pattern, hh + 1, ii + 1, jj + 1))
                    filename = os.path.join(targetfolder, helper)
                img_png = img_png.transpose()
                img = cv2.flip(img_png,0)
                if img.sum() != 0:
                    cv2.imwrite(filename,img_png)
                    if printfilename:
                        print(filename)
                else:
                    if printfilename:
                        print("Empty Slice Skipped => \t", filename)
    end = time.time()
    print("Time :", end-start)


###############################################################################
def main(parserargs):
    ff = r'{}'.format(parserargs.inputfolder)
    pref = parserargs.prefix
    pt = parserargs.pattern
    targetfolder = r'{}'.format(parserargs.targetfolder)
    imgconverting(ff, targetfolder, pref, pt, slicedrop=parserargs.drop, printfilename=parserargs.verbose)


###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfolder", required=True, help="Input folder containing nii.gz files")
    parser.add_argument("-o", "--targetfolder", required=True, help="Target folder to store PNG slices")
    parser.add_argument("-p", "--prefix", required=True, help="Prefix for PNG slices")
    parser.add_argument("-t", "--pattern", default=False, help="String pattern as suffix for PNG slices. Default is False : original file name is kept")
    parser.add_argument("-d", "--drop", type=int, default=10, help="Number of slices to drop from the beginning of Nifti. Default is 10 slices")
    parser.add_argument("-v","--verbose",type=bool, default=True, help="Print file names. Default is True")
    args = parser.parse_args()
    main(args)
