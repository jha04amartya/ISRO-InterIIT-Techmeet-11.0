# ISRO-InterIIT
## ISRO PS Submission

This repo is the main submission of the Chandrayaan PS by ISRO for INTER-ITT Tech Meet 11.0 . We have tried to address both the problems mentioned in the main problem statements as mentioned below:

- Development of an AI/ML model to generate high (~30 cm) resolution lunar terrain image from medium/low (5 m / 10 m) resolution terrain image, and evaluate its accuracy; the model will be amenable to further training and improvement;
- To generate a global lunar atlas (digital) with the help of the AI/ML model, based on the medium / low resolution data available.
The AI/ML model to be developed by using the publically available overlapping data of OHRC (~30 cm resolution) and TMC-2 (5 m /10 m resolution) payload data on board Chandrayaan-2 Orbiter.

## Our Aproach
Our approach was divided into solving into tasks:

### Generating the Dataset
A dataset containing the images of overlapping areas from both TMC and OHRC instruments was created by scrapping the data from ISRO Pradhan Website. This task was accomplished by using both scripts written to scrape the website, as well as visual matching to generate patches. Once TMC and OHRC images were downloaded after matching, metadata for each image was extracted by parsing the corresponding XML file, to get the information about the coordinates, pixel resolution etc. Then, we converted the .img files and compressed and saved them to .npz format. This resulted in a size reduction of 60%, without losing quality. This resulted in 2 datasets, one containing OHRC images and another containing TMC images.

### Matching TMC to OHRC Images
First we downloaded the downsampled version of the datasets, and extracted their corner coordinates. Using custom scripts to calculate HaverSine distances, we matched the OHRC image corners which were contained in the TMC image corners. This resulted in about 18 patches of TMC and OHRS images, of which only 6 TMC were unique. Hence, we only had to download high resolution version of 6 TMC and 18 OHRC images. Code regarding this can be found at 'image_processing/ohrc_tmc_overlap.py'

### Image Processing
First, we interpolated coordinates for each pixel around the corner coordinates of OHRC in TMC image, to accurately get the coordinates of OHRC in TMC. Afterwards, we defined a bounding box around the common patch and masked the remaining area. Affine transformations were applied to align the images correctly. Code regarding this can be found in the file 'image_processing/ohrc_tmc_preprocessing.ipynb'


### Training and Evaluating the Model
Different models were identified as potential solutions to our problem, which included SRCNN, ESRGAN, Latent Diffusion models, VAE etc. Out of these, ESRGANs performed the best for satellite imagery, hence we decided to move formward with this.
Thus, we trained the ESRGAN( Enhanced Super-Resolution Generative Adversarial Networks) using the standard resolution training dataset, and then performed transfer learning to finetune it over the the matches between TMC and OHRC. We got 16 complete matches between OHRC and TMC and 2 partial matches which were used to finetune the model. The metrics used were SSIM and PSNR, calculated by matching the images produced by our model and given OHRC images. 
Our model achieved an average performance of 0.77 on SSIM and 15 on PSNR.

The trained model was then stored effieciently in the form of an '.pb' file using the TensorFlow SaveModel features, which allows storing the model architecture, the weights and all other variables related to the model in one compressed file.

### Creating the Map - The Lunar Atlas
After tackling the 1st part of the given problem statement successfully, we moved on to solving the 2nd part, i.e. creating a Lunar Atlas. For this, we wrote a script for creating a map of the whole lunar surface using the existing TMC images and use it to generate 0.5 degrees X 0.5 degrees latitude and longitude patch which was passed through our model to create a high resolution part of the map. A part of the map with corner coordinates at : 
top left     : -68.36261 ,  41.141334
top right    : -68.35982 ,  41.420493
bottom left  : -69.194672,  41.482045
bottom right : -69.197495,  41.19252 

can be found at the following drive link : https://drive.google.com/file/d/1yck1UF5hzbotBJg8LTPnmOHqVz3L217J/view?usp=share_link

## Using our Model
The model can be executed by following these steps:

- The demonstration will be done through the 'executable.ipynb' IPython notebook located in the folder 'demonstration'. 
- Store the TMC images and corresponding OHRC images in the same directory ('demonstration') as the model file. Make sure that the files are correctly formatted with each image having its corresponding XML file.
- For demonstration we have compressed and loaded the binary images as NumPy arrays and saved them in a drive link at
- In the 'executable.ipynb' file, please put the path of the image you want to upscale. The are where path has to be added
has been highlighted through comments
- The default path has been set as the np array paths of the files you will download from the drive.
- The trained '.pb' model has been stored in folder 'demonstration/model' please do not change this, else the trained model wont load.
- Run the cells of executable.ipynb and follow along the instructions provided there.
- The output will be saved as "final_image.tif"