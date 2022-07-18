# Immunostain Image Analysis

This image analysis program was conceived as a means to streamline the tasks of
de-noising and emphasizing regions of interest (ROIs) in images acquired via 
immunofluorescence imaging. Its primary function is to take a 
user-provided red-green-blue (RGB) fluorescence image or image folder input, 
isolate a specific stain channel to be examined, optimize the image(s) for analysis 
through a series of processes including noise filtration and selective exclusion 
of non-ROIs, and then collect and compile data points of interest from the 
post-processing images, from which tangible and, ideally, reliable results can be
drawn.

## Capabilities
At the moment, the program works very reliably when provided with the requisite
user inputs, including an image file pathway, the stain channel one wishes to
create a mask of, and the nucleus size threshold defining the minimum pixel cluster 
area for which objects - in this case cells - will be included in the data 
analysis. 

(Insert images here?)

## Future Development
Future developments of the program are currently being directed towards expanding 
the options available to the user.
- A more traditional image filtering pathway is
  currently in development, as an optional - albeit not necessarily superior - 
  compliment to the machine learning approach currently in use. This approach will
  involve the implementation of a Gaussian filter for image denoising, followed by 
  a 0-100 scale thresholding system to replicate the nucleus size threshold
  currently in use.
- Another concept that has been discussed is the implementation of automated
  minimum pixel area determination, in which the program would be able to
  determine a good cell area threshold independent of user input. This could 
  possibly be done by having the program identify the lense magnification used 
  for the image(s) run through it, and have that as a basis for minimum cell area.
- Addressing remaining challenges is also a consideration, such as introducing
  some means of accurately separating cells that are in contact.


## Methodology/Algorithm(s)
In its current form, this program consists of an entirely Python-based backbone
for its interface, object management, and practical applicability of the 
algorithm(s) used. The code was written in both Miniconda and Pycharm environments,
with a variety of different function packages being utilized as outlined in the 
separate requirements.txt file.

The actual image analysis is performed by a U-Net neural 
network, which has been trained on immunofluorescence images to be able to
accurately identify ROIs (in this case mesenchymal stem cells) and isolate them
from undesired objects (often debris or other non-stem cells mixed in with the
experimental batch). Those defined ROIs then become the areas from which various
data is taken - namely stain pixel intensity and distribution - and quantified
numerically in the form of an Excel spreadsheet.

As was mentioned previously, future iterations of this program will likely include
more algorithms, specifically a Gaussian filter and possibly even more filter
implementations as an alternative to the neural network approach.


## How to Use
1) Download and install all necessary applications and components. This includes:
   1) Java (8)?
   2) Visual Studio**
   3) Git Bash
   4) Python 3
   5) PyCharm (Community Edition)
   6) ZEN blue
   7) Image J
   8) Miniconda
- ** It is recommended to install as many C++ addons as available, to avoid
  future complications
2) Create a code environment using either Miniconda or PyCharm. Either one should
   work, but some technical difficulties have been noted with Miniconda, so PyCharm
   may be the safer choice for a 1st-time setup.
3) Download this code and open it in your new environment.
4) Install all necessary python packages. Any packages that you're missing will
   be underlined in red in the code files they're imported in. The main ones to
   look out for are:
   1) javabridge
   2) bioformats
   3) opencv
   4) numpy
   5) pytorch
   6) pil
   7) unet
5) ***If using unet, download the unet program and place it in a "models" folder
   inside the "unet" folder that comes with the code
   - ***It is not currently possible to download the specific neural network we use
     outside of our lab group
6) Once all packages and applications are successfully installed, and the 
   environment has been properly set up (if you need help, there are numerous online guides on how to
   do this), the code should be just about ready to run. All that's needed are
   the requisite user inputs in the main.py file:
   1) For "bioformat_imgs_path =", provide a string of the location on your PC
      from where images will be drawn
      - EX: r"D:\BioLab\img\Images for matlab quant\10x"
   2) For "unet_model =", if using a unet algorithm for analysis, provide a string 
      of the location within your Immunostained_Image_Analysis folder where your 
      unet file is located
   3) For "nucl_area_min_pixels_num =", you may adjust the number value according
      to whatever size of pixel cluster you would like to establish as the minimum
      for data consideration. One may want to play around with this feature to get
      a sense of the scale.
   4) (COMING SOON) For "nuc_recognition_mode =" input either "unet" to run images
      through the machine learning algorithm or "thr" to run them through the
      conventional filtering algorithm
7) If all of the above parameters are to your liking, you may run the main.py file,
   which should allow the code to run assuming there are no complications. As long
   as the "bioformat_imgs_path =" points towards a valid image location, all
   images there will be run through and analyzed. From the originals, temporary
   post-analysis images will be made with contours drawn around objects that the
   program identifies as cells, which can be found in the nuclei_area_verification
   folder under analysis_data. Lastly, the program will compile numerical data
   from the images into an Excel file labeled as "stat", also found in the 
   analysis_data folder.


## FAQ
    
-Will do later

## The Team
-Benjamin Morenas

-Nina Nikitina

-Gunes Uzer

-Sean Howard

### May want to include images?
