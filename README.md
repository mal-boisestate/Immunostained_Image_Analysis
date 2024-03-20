<img width="583" alt="segmentation" src="https://github.com/mal-boisestate/Immunostained_Image_Analysis/assets/107217268/e69a7ab2-1196-44f7-aa96-2e1fe03c5d85">

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
user inputs, including an image file pathway, the output file location, the stain channel one 
wishes to create a mask of, and the nucleus size threshold defining the minimum pixel cluster 
area for which objects - in this case cells - will be included in the data 
analysis. 


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


## How to Use
Please refer to the pdf labeled "Documentation" in this project for detailed instructions regarding program features, installation, and usage.


## The Team
Benjamin Morenas, Nina Nikitina, and Gunes Uzer
