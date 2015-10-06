import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import time
from evolution import *
from plot_u import *
import os, sys
import Image
import ImageEnhance
import ImageFilter
import ImageDraw
import ImageOps
import scipy.ndimage
import pickle

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="image file to read")
parser.add_option("-o", "--output", dest="output",
                  help="output directory")
parser.add_option("-r", "--resized", type="int", dest="res_size",
                  help="image size after resizing")
                  
parser.add_option("-k", "--kernel", type="int", dest="kernel_size",
                  help="image size after resizing")

parser.add_option("-t", "--threshold", type="int", dest="threshold",
                  help="image size after resizing")
                  
parser.add_option("-e", "--epsilon", type="float", dest="epsilon",
                  help="epsilon in level set algorithm")

parser.add_option("-s", "--timestep", type="int", dest="timestep",
                  help="timestep in level set algorithm")
                  
parser.add_option("-l", "--lambda", type="float", dest="lam",
                  help="lambda parameter for lset segmentation")

parser.add_option("-a", "--alpha", type="int", dest="alpha",
                  help="alpha parameter for lset segmentation")

parser.add_option("-n", "--num", type="int", dest="num",
                  help="number of iterations")

parser.add_option("-i", "--inverse",
                  action="store_true", dest="verbose", default=False,
                  help="invert image")
                  
parser.add_option("-v", action="store_true", dest="verbose")
 

(options, args) = parser.parse_args()


def gauss_kern():
    """ Returns a normalized 2D gauss kernel array for convolutions """
    h1 = 15
    h2 = 15
    x, y = np.mgrid[0:h2, 0:h1]
    x = x-h2/2
    y = y-h1/2
    sigma = 1.5
    g = np.exp( -( x**2 + y**2 ) / (2*sigma**2) );
    return g / g.sum()
    
    
filename = options.filename
print filename, options.alpha
im = Image.open(filename)

outputDir = options.output

        
print "loading image ", filename, "... " ,

enhancer = ImageEnhance.Color(im)

imGray = enhancer.enhance(0.).convert("L").resize((options.res_size,options.res_size), Image.ANTIALIAS).filter(ImageFilter.MedianFilter(options.kernel_size))

print "done."

print "applying threshold... ",

hist = scipy.ndimage.filters.gaussian_filter1d(np.array(imGray.histogram()[100:200]), sigma=10)
val = np.argmin(hist) + options.threshold

contrast = ImageEnhance.Contrast(imGray)
imGray = contrast.enhance(2.5)

sharper = ImageEnhance.Sharpness(imGray)
imGray = sharper.enhance(0.5)
imGray.save("temp.bmp")
im.convert("L").resize((options.res_size,options.res_size), Image.ANTIALIAS).save("temp2.bmp")
Img = plt.imread("temp.bmp")[::-1]
#Img -= val
#Img = Img.clip(min=val)

Img2 = plt.imread("temp2.bmp")[::-1]

plt.imshow(Img, cmap='gray')
plt.draw()
plt.savefig("temp6.png")
plt.hold(False)

if options.verbose: 
	plt.ion()

g = gauss_kern()
Img_smooth = signal.convolve(Img,g,mode='same')
Iy,Ix=np.gradient(Img_smooth)

f=Ix**2+Iy**2
g=1. / (1.+f)  # edge indicator function.

epsilon = options.epsilon #1.5  the papramater in the definition of smoothed Dirac function
timestep = options.timestep #5 time step
mu = 0.2 / timestep  # coefficient of the internal (penalizing) 
                  # energy term P(\phi)
                  # Note: the product timestep*mu must be less 
                  # than 0.25 for stability!

lam = options.lam #10.  coefficient of the weighted length term Lg(\phi)
alf = options.alpha #-3  coefficient of the weighted area term Ag(\phi);
      # Note: Choose a positive(negative) alf if the 
      # initial contour is outside(inside) the object.

nrow, ncol=Img.shape

c0=-4

u=c0*np.ones((nrow,ncol))

w=8

initialu = Img
initialu /= 32.
u += initialu
#try:
num = int(filename[len(filename)-8:len(filename)-4]) - 1
if num > 0 and not (num > 260 and num <263):
	newname = filename.split("/")[-1].split(".")[0]
	num = "0" + str(num)
	newname = newname[:len(newname)-len(num)] + num
	print outputDir + newname + "_seg.png.con"
	u += pickle.load(open(outputDir + newname + "_seg.png.con", "rb"))
	u /= 2
#except:
#	print "failed to load data!"
#	pass



#plot_u(u)

for n in range(options.num):    
    u=evolution(u, g ,lam, mu, alf, epsilon, timestep, 1)
    
    if options.verbose and np.mod(n,10)==0:        
        #plot_u(u)
        plt.imshow(Img2, cmap='gray')
        plt.draw()
        plt.hold(True)
        CS = plt.contour(u,0, colors='r') 
        plt.draw()
        plt.hold(False)

plt.imshow(Img, cmap='gray')
plt.draw()
plt.hold(True)
CS = plt.contour(u,0, colors='r') 
plt.draw()
plt.hold(False)
    
outputName = outputDir + filename.split("/")[len(filename.split("/"))-1].split(".")[0] + "_seg.png"

print "saving output in ", outputName ,"... ",
plt.savefig(outputName)
pickle.dump(u,open(outputName+".con", "w+"))
print "done."
