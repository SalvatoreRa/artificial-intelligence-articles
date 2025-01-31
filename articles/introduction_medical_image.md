# Introduction to medical image analysis

## Index

* [Introduction to medical imaging](#Introduction-to-medical-imaging)
  * [Modalities](#Modalities)
  * [Framework](#Framework)
  * [Deep learning in medical image analysis](#Deep-learning-in-medical-image-analysis)
  * [Summary of the section](#Summary-of-the-section)
* [Introduction to point processing](#Introduction-to-point-processing)
  * [Histogram](#Histogram)
  * [Point preprocessing](#Point-preprocessing)
* [Introduction to Thresholding](#Introduction-to-Thresholding)
  * [Thresholding classification](#Thresholding-classification)
  * [Thresholding](#Thresholding)
  * [Automatic thresholding](#Automatic-thresholding)
  * [Adaptative thresholding](#Adaptative-thresholding)
* [Neighborhood image processing](#Neighborhood-image-processing)
  * [Help thy neighbors](#Help-thy-neighbors)
  * [Neighborhood preprocessing](#Neighborhood-preprocessing)
  * [Correlation or convolution](#Correlation-or-convolution)
  * [Edge detection](#Edge-detection)
  * [Canny edge detection](#Canny-edge-detection)
* [Morphological image processing](#Morphological-image-processing)
  * [Introduction to morphology](#Introduction-to-morphology)
  * [Morphology operations](#Morphology-operations)
  * [A little practical example](#A-little-practical-example)
* [A Practical Guide to BLOB Analysis and Extraction ](#A-Practical-Guide-to-BLOB-Analysis-and-Extraction)
  * [What is a Blob?](#What-is-a-Blob)
  * [Blob Detection and Extraction](#Blob-Detection-and-Extraction)
  * [The Grass-fire algorithm](#The-Grass-fire-algorithm)
  * [BLOB Features](#BLOB-Features)
  * [BLOB classification](#BLOB-classification)
  * [Hunting for panda](#Hunting-for-panda)
* [Harnessing the power of colors](#Harnessing-the-power-of-colors)
  * [What is a color?](#What-is-a-color?)
  * [Color images](#Color-images)
  * [Dramatic black and white](#Dramatic-black-and-white)
  * [Counting color in an image](#Counting-color-in-an-image)
  * [Coloring black and white images](#Coloring-black-and-white-images)
* [Image Segmentation](#Image-Segmentation)
  * [Pixel classification](#Pixel-classification)
  * [The parametric approach or the Gaussian triumph](The-parametric-approach-or-the-Gaussian-triumph) 

 
All the code about these tutorials is stored [here](https://github.com/SalvatoreRa/tutorial)

## Introduction to medical imaging

In general, we can say that the primary scope of radiological imaging is to produce images, which depict anatomy or physiological function well below the skin surface.

Different types of medical images are produced by varying the types of energies used to acquire the image. These different modes are called [radiology modalities](https://ccdcare.com/resource-center/radiology-modalities/). Different modalities present different aspects (time of exposition, scanning methods, use of radioactive isotopes). As an example, we prefer images that can be acquired in a short time. However, sometimes this is not possible, like in nuclear medicine where you use [radioactive isotypes](https://world-nuclear.org/information-library/non-power-nuclear-applications/radioisotopes-research/radioisotopes-in-medicine) (you need to inject them in the patient and wait for them to diffuse, and since the time of decay is decided by the physics it can require minutes to achieve an image). A slower method has its drawback: the patient has the involuntary motion of the lung, heart, and esophagus over this time frame, thus long time to scan leads to a lower resolution.

The goals of medical image analysis are:

* **Quantification**, measuring the features of a medical image (like area or volume)
* **Segmentation**, which is a step used in general to make features measurable (you segment an object and you measure the properties)
* **Computer-aided diagnosis**, given measurements and features makes a diagnosis.

### Modalities

* **[Radiography](https://en.wikipedia.org/wiki/Radiography)**. It was the first medical imaging technology, thanks to the physicist Wilhelm Roentgen who discovered X-rays on November 8, 1895 (he made the first radiography of the human body when produced an image of his wife’s hand). Radiography is performed with an X-ray source on one side of the patient and an X-ray detector on the other side (a short-duration pulse of X-rays is emitted by the X-ray tube, it passes by the patient and arrives on the detector producing the image). The interaction with the patient is scattered and that is recorded in the image (this phenomenon is called attenuation). The attenuation properties of the anatomic structures inside the patient such as bone, liver, and lung are very different, allowing us to have an image of these tissues.
* **[Mammography](https://www.nibib.nih.gov/science-education/science-topics/mammography)** is a radiography of the breast. Since the low subject contrast in breast tissues, the technique uses much lower x-ray energies than radiography, and thus the x-ray and detector systems are different and specifically designed for breast imaging. It is used to routinely evaluate asymptomatic women for breast cancer.
* **[Computer tomography](https://en.wikipedia.org/wiki/CT_scan)** was a breakthrough in medicine during the 70s eliminating the use of exploratory surgery. CT images are obtained by acquiring numerous (around 1000) X-ray projection images over a large angular swath by the rotation of the X-ray source and detector. The acquired images are then reconstructed with an algorithm. CT results in high-resolution thin-slice images of an individual (moreover allowing a 3D reconstruction). Moreover, there is no superimposition of the anatomical structure allowing a better interpretation. The CT image data set can be used to diagnose the presence of cancer, ruptured disks, subdural hematomas, aneurysms, and other pathologies. There is a balance between radiation dose and image resolution, mathematical techniques and artificial intelligence are helping to increase the resolution keeping the same dose. In addition, the introduction of contrast can be used to study vascularity and perfusion of organs
* **[Magnetic resonance imaging](https://en.wikipedia.org/wiki/Magnetic_resonance_imaging)**. MRI uses magnetic fields that are about 10,000 to 60,000 times stronger than the Earth’s magnetic field. MRI utilizes the nuclear magnetic resonance (NMR) properties of the nucleus of the hydrogen atom.
* **[Fluoroscopy](https://en.wikipedia.org/wiki/Fluoroscopy)**. It uses X-ray detector systems capable of producing images in rapid temporal sequence. Fluoroscopy is used for positioning intravascular catheters, visualizing contrast agents in the GI tract, and image-guided intervention including arterial stenting.
* **[Ultrasound](https://www.mayoclinic.org/tests-procedures/ultrasound/about/pac-20395177)**. It uses high-frequency sound waves, that are reflected off tissue to develop images of joints, muscles, organs, and soft tissues. It is considered the safest form of medical imaging and is used in a wide range of cases
* **[Positron emission tomography](https://en.wikipedia.org/wiki/Positron_emission_tomography)**. PET is a functional imaging technique that uses radioactive substances known as radiotracers to visualize and measure changes in metabolic processes. It is extensively used in oncology to research metastasis. However, it is also used in brain imaging for research on seizures.

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/image_medical.webp)
*from Wikipedia*

### Framework

In medical images, there have been proposed different frameworks, but we could define them as a general framework:

* Image acquisition encompasses everything related to image acquisition (camera parameter, light, and so on).
* In pre-processing, the initial processes need to use the image in your model
* Segmentation, where the relevant part of the image is separated and extracted
* Representation, the extracted feature is represented in a more concise manner
* Recognition, where the feature is analyzed and classified.

**Image acquisition**

Two parameters are often important

* [Spatial resolution](https://en.wikipedia.org/wiki/Spatial_resolution): number of [pixels](https://en.wikipedia.org/wiki/Pixel) for representing an image
* [Gray level quantization](https://en.wikipedia.org/wiki/Quantization_(image_processing)) is the number of gray levels to represent the image. Typical gray resolutions are 8, 10, and 12-bit (8-bit is corresponding to 256 levels). In medical images, there is an associated physical measurement to these gray levels (for instance in computer tomography, a pixel value represents the Hounsfield Units (HU))

**Digital images**

The image’s content is transformed into a pixel (with a value range from 0 to 255, defined as the intensity of the pixel). A [gray-scale image](https://en.wikipedia.org/wiki/Grayscale) is just a 2D matrix with m x n pixels (in this case 0 represents black and 255 pure white). A color image is defined by three matrices (one for each channel in the RGB format) and therefore the format of m x n x c. The total number of pixels (m x n) defines the size of the image (for instance for a sensor this is the maximum image it can acquire). We can have also multi-spectral images or multi-channel images (ex. Satellite images that are integrated with other wavelengths like infrared).

A special case is **[binary images](https://en.wikipedia.org/wiki/Binary_image)**, where 0 represents the background and 1 the foreground (normally, these images are the results of thresholding and other algorithms). Another special case is label images, where pixels are associated with a number, representing to which object they belong (for example a label image can have a value of 0 for the sky, 1 for each pixel part of a human figure, 2 for vehicles, and so on). Normally, Label images are obtained after image segmentation (or hand-annotated as an example to train these algorithms).

Generally, in programming languages, the image is converted to an array (a matrix m x n) and the first pixel has generally coordinates 0,0 (starting from the upper left corner). These coordinates are used also in the plotting. Programs are conducting mathematical operations on each point of the matrix (pixel), with large images this can be computationally expensive.

Moreover, not all the image is interesting for further analysis, it is often common to define a **[region of interest (ROI)](https://en.wikipedia.org/wiki/Region_of_interest)** which is generally a common rectangle containing the pixel of interest. As a simple example, if you want to count the daily car numbers in the mall’s parking over one year, your ROI in the image would be the zone encompassing the parking area.

**Image compression**

An image without compression (256 x 256) pixels with 1 byte (256 levels) has a size of 65 kb. But raw images can be extremely large and a dataset can be composed of thousands of images (with a large cost of storage). A [compression algorithm](https://en.wikipedia.org/wiki/Data_compression) aims to represent an image with fewer pixels. Since many neighbor pixels present the same color (or sale value) an image can be easily compressed. This is the principle of the simplest compression algorithm [Run-Length Encoding (RLE)](https://en.wikipedia.org/wiki/Run-length_encoding), where for the pixels with the same values we store only the value and the count of the pixels with the same value. After running the algorithm we can calculate the compression ratio (uncompressed size/ compressed size). Notice, that RLE does not lead to loss of information (or lossless compression). But not all the compression algorithms are lossless, while PNG is lossless [JPEG](https://en.wikipedia.org/wiki/JPEG) is lossy.

Choosing the compression level is important because high compression leads to artifacts in the image. Indeed, the lossy image format should be avoided so as to not introduce artifacts in the analysis.

**DICOM format**

A common image format for medical imaging (it contains additional information in the header such as hospital, patient, scanner, image information, and anatomy axis). For CT or PET scans, each slice is an image and the patient can be scanned multiple times ([DICOM](https://en.wikipedia.org/wiki/DICOM) format contains information about the series). [PyDicom](https://pydicom.github.io/) is a popular package for analyzing DICOM images in Python. It can be lossy or lossless.

### Deep learning in medical image analysis

[Artificial intelligence (AI)](https://en.wikipedia.org/wiki/Artificial_intelligence) and [machine learning](https://en.wikipedia.org/wiki/Machine_learning), especially deep learning, applications are used extensively in everyday life and in many domains (from cars to internet apps). There is extensive literature about deep learning and medical image analysis since this is an active field of research. However, there are not so many algorithms that are used in hospitals and healthcare. In fact, there is a lot at stake in the medical imaging field, since an algorithm can have serious consequences (missing a cancer diagnosis, leads to delayed treatment, and potential loss of life).

Just as a brief introduction, the difference between [deep learning](https://en.wikipedia.org/wiki/Deep_learning) and traditional machine learning techniques is that the former can automatically learn meaningful representations of the data (this means you do not need a lot of preprocessing to extract features by yourself). However, deep learning requires a lot of data, is computationally expensive, and requires sophisticated hardware. Moreover, many deep learning models have been trained and tested on small images (64x64, 256x256) while medical images are often in high resolutions (or we can 3d or even 4d). In addition, the medical images training set needs to be annotated by a medical expert who will establish the ground truth (which means a lot of work from an expert if your dataset is quite large). There are also problems with artifacts, different machines used to acquire the images, and so on.

There are many tasks and challenges where deep learning can be useful in medical image analysis. It is interesting to discuss a few of them just as an introduction.

For example, prior to deep learning, most [medical image segmentation](https://paperswithcode.com/task/medical-image-segmentation) methods relied on algorithms like thresholding, clustering, and other human-designed algorithms. Deep learning instead needs just an example to generate its own algorithm and to discriminate and proceed to segmentation. Indeed, segmentation is a tedious and long process to conduct by hand, thus image segmentation is an active field of research and playground for deep learning. Image segmentation is important to select potential cancer lesions, and metastasis, study blood vessels, and so on. A different but similar task is object localization (there are two kidneys in the body, segmentation is to select the kidney pixels while localization is to identify the two different organs). Moreover, the extraction of features ([radiomics](https://en.wikipedia.org/wiki/Radiomics)) can be fundamental to creating predictive models.

However, another important challenge in the use of deep learning in clinics is that often these models are a black box. Explainability is fundamental to use in daily healthcare, we need to know why a model is predicting a certain outcome, or what drives its decision when identifying in the image a region as abnormal.

### Summary of the section

The world of biomedical imaging is a fascinating one, but it also presents complex challenges. Given the importance of these types of data for both diagnostics and countless applications, countless groups and models have been developed. In subsequent articles, we will address the basics of being able to analyze images.

## Introduction to point processing

When you have used an Instagram filter to correct brightness, without noticing you are doing point processing. In simple words, you have an input image f(x,y) and you want to produce an output image g(x,y). In point processing, we are conducting an operation on the image in the way a pixel now has a new image. In comparison to other processing functions, here the value of the pixel in the output is based on the value of the pixel in the input image. The name point processing derives from the fact that the neighbor pixels have no role in the transformation.

### Histogram

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/Example_histogram.png)
*from Wikipedia*

A [histogram](https://en.wikipedia.org/wiki/Histogram) is a graphical representation of the data that organizes a group of data points into user-specified ranges. In a way, looks like a bar graph, but differently from the bar graph is a more sophisticated plot that condenses a data series: grouping data into logical ranges or bins. Histograms are primarily used in statistics, for instance in demography someone can use a histogram to show how many people are between a certain age range (0–10, 11–20, 21–30, and so on). The **range of bins** is decided by the user: we can divide by an arbitrary number of bins (for instance, in the above example we used bins of width 10, but we could have decided the interval was 5 years).

In general, we plot the **frequency** on the y-axis (if we have 655 individuals in the interval 0–10, the corresponding bar would have a value of 655). However, we can plot the **percentage of the total or density**. Despite many people using histograms and bar charts as interchangeable terms, histograms provide the frequency distribution of variables while bar charts are more indicative to represent a graphical comparison of discrete or [categorical variables](https://en.wikipedia.org/wiki/Categorical_variable). In fact, histograms are in general used with continuous variables.

Histograms have shown to be valuable in many cases: different types of data, time series where you group data for hours and show the frequency on the y-axis, and also images. Histograms are very useful when you are interested in the general distribution of your features (to check if the distribution is [skewed](https://en.wikipedia.org/wiki/Skewness), symmetric or there are [outliers](https://en.wikipedia.org/wiki/Outlier)).


A few suggestions for using histograms:

* Variables should be continuous since a histogram is meant to describe the frequency distribution of a continuous variable and thus is a misuse if you use a categorical variable.
* Use a **zero-valued baseline**, since the height of each bin represents the frequency of the data changing the baseline (starting from an arbitrary height, insert a gap) is altering the perception of the data distribution
* The choice of the number of bins is important. In general, there are some rules of thumb but domain knowledge is normally driving the choice (and better try some options and see which one is leading to the best result). Remember that the bin size is normally in inverse relationship with the number of bins (the larger the bins, the fewer cover the data distribution, however choosing too many bins is a bad choice as well). In fact, if you have too many bins the data distribution looks rough (and hard to understand) but too few, you do have not enough details. Many histogram implementations use some algorithms to find the appropriate number of bins.
* Bin size can be also of unequal size in case of sparse data (but this is dangerous water and you should avoid it)

**Image histogram**

The [image histogram](https://en.wikipedia.org/wiki/Image_histogram) (value of the pixels on the horizontal axis and frequency on the vertical axis) allows for inspecting the image. The histogram represents the tonal distribution of a digital image (the number of pixels in an image in the function of their intensity), allowing us to inspect the distribution. If most of the pixels are close to the left, the image would be dark. Conversely, if the curve is too close to the right edge, the image will be too bright. Histogram peaks can be informative to retrieve a particular structure in the image and can be used to select them through the threshold. Moreover, can be used to adjust contrast, brightness, or tonal value.

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/image_instogram.webp)

The histogram of an image can be described as a function h(v) where h is the frequency of pixels with value v, where the total pixel number is N

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/image_instogram2.webp)

Which can be normalized by dividing by N

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/image_instogram3.webp)

In this case, H(v) represents the probability for a pixel to have value v:

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/image_instogram4.webp)

The [cumulative histogram](https://www2.tulane.edu/~salem/Cumulative%20Histogram.html) is the progressive sum for each bin of the pixel frequencies (if for a histogram the frequencies for the first 3 bins are 2, 3, 6 for the cumulative is 2, 5, 11). More formally. For a histogram with n bins (an arbitrary number between 0 and 255) with height H(n) the cumulative histogram:

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/image_instogram5.webp)

I will show here how to load an image with [Python](https://www.python.org/) (there are many ways, but this is easier with [Google Colab](https://colab.research.google.com/)). The image is also resized to be easier to handle.

```python 
#loading image from url
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.transform import rescale
from skimage import  color
import scipy.ndimage as ndi

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/AP_lumbar_xray.jpg/255px-AP_lumbar_xray.jpg"
a = io.imread(url)

im = color.rgb2gray(a)
#rescaling
a = image_rescaled = rescale(im, 0.5, anti_aliasing=False)
plt.imshow(a, cmap = "gray")
plt.axis('off')
plt.show()
```

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing.webp)

Here plotted the histogram and CDF

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing2.webp)

Here the image is in black and white, but you can use a histogram also for color images. In the second case, you can use also a separate histogram for each channel.

### Point preprocessing

A point preprocessing is a function that takes into account an input image f(x, y) and returns an output image g(x, y). In concrete, applying this function you change the value of each pixel without considering the neighbor pixels.

**Gray level mapping**

**Brightness modification** is changing the value of each pixel by adding a constant c (or subtracting)

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing3.webp)

Since the range is 0–255 if after the transformation some pixels will have a value over 255 they will be set to 255. In the case of subtracting, if some pixels have values less than 0, they will be set to zero. In concrete looking at the histogram of the function, we can notice that brightness processing is shifting the function.

In Python is very simple:

```python 
def brightness_modification(image= None, c= 0.0):
  """brightness correction of im
     image as numpy array
     image range in 0-1
  """
  image = image + c
  image = np.where(image >1,1, image) #one is the max
  image = np.where(image <0,0, image)
  return image
```

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing4.webp)

**[Contrast correction](https://www.mathworks.com/help/images/contrast-adjustment.html)** is meant to increase the separation of pixels with close value by multiplying for a constant c. In this case, the slope of the function is modified.

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing5.webp)

Which we can combine as:

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing6.webp)

in Python:

```python 
def contrast_modification(image= None, c= 0.0):
  """contrast correction of im
     image as numpy array
     image range in 0-1
  """
  image = image * c
  image = np.where(image >1,1, image) #one is the max
  image = np.where(image <0,0, image)
  return image
```

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing7.webp)

Some algorithms require that the image is in a certain range (ex. 0–1). In this case, we have to **remap to a different range the pixels**. This is the general equation, where Vmax and Vmin are the actual maximum and minimal values of the image, and Vmax’ and Vmin’ are the desired extremes of the new range:

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing8.webp)

For the transformation to the interval 0,1 and the inverse transformation:

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing9.webp)

Another transformation is **linear mapping**, considering the mean µ and the standard deviation σ we can map the input image to a new mean µ’ and the standard deviation σ’:

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing10.webp)

In Python:

```python 
def linear_mapping(image= None, std= 0.0, mean = 0.0):
  """linear mapping of im
     image as numpy array
     image range in 0-1
  """
  im_mean = np.mean(image)
  im_std = np.std(image)
  image = ((mean/im_mean)*(image -im_std))+std
  image = np.where(image >1,1, image) #one is the max
  image = np.where(image <0,0, image)
  return image
```

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing11.webp)

**[Histogram stretching](https://en.wikipedia.org/wiki/Histogram_equalization)** is a technique to adjust contrast and brightness at the same time. The idea is to stretch the image histogram so very dark and very bright bins are used (i.e. near 0 and near 255). Considering a histogram with the minimum value of f1 and the maximum value of f2 (ex. An image with a pixel value range 20–180 has f1= 20 and f2 = 180), histogram stretching can be obtained:

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing12.webp)

For our example:

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing13.webp)

Histogram stretching enhances contrast and enables better recognition of small details, however, is sensible to outliers. Therefore, a more sophisticated technique of **[histogram equalization](https://en.wikipedia.org/wiki/Histogram_equalization)** can be used. This transformation is based on the cumulative histogram.

There are also derived algorithms as **[adaptative histogram equalization](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization)** and **Contrastive Limited Adaptive Equalization** which avoid noise amplification.

```python 
from skimage import data, img_as_float
from skimage import exposure

matplotlib.rcParams['font.size'] = 12
# Load an example image
img = im

# Contrast stretching
p2, p98 = np.percentile(img, (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

# Equalization
img_eq = exposure.equalize_hist(img)

# Adaptive Equalization
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
```

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing14.webp)

**Non-Linear Gray-Level Mapping**

**[Gamma mapping](https://en.wikipedia.org/wiki/Gamma_correction)** is defined by elevating to γ constant the f(x,y). The value of γ < 1 increases the dynamics in dark areas (the mid values are increased, basically the gray pixels are affected), why γ > 1 increase the bright area (the mid pixel values are decreased)

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing15.webp)

**Logarithmic mapping** is using the [logarithmic function](https://en.wikipedia.org/wiki/Logarithmic_scale) to enhance pixels with low intensity. Generally, it is used when there are few bright spots on a dark background or when the dynamic range is large.

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing16.webp)

Since logarithms are not defined at zero we are adding one and we add c in the equation to be sure the max output is 255

**Exponential mapping** is increasing details in the light areas while decreasing in the dark ones. We can tune the transformation using a parameter c:

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing17.webp)

**Sigmoid mapping** is also used, showing interesting results, the applied mapping is:

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing18.webp)

```python 
# Gamma
gamma_corrected = exposure.adjust_gamma(img, 2)
# Logarithmic
logarithmic_corrected = exposure.adjust_log(img, 1)

# sigmoid
sigmoid_cor = exposure.adjust_sigmoid(img)
```
![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing19.webp)

The associated code of this article can be found **[here](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/Point_processing.ipynb)**.

## Introduction to Thresholding

**[Thresholding](https://en.wikipedia.org/wiki/Thresholding_(image_processing))** can be defined as the simplest method of segmenting images in digital image processing. This can be useful in many contexts, like when you want to separate the foreground from the background. In general, we are interested in the foreground since often the background contains some noise. Notice that when you are applying the thresholding to a grayscale image you are obtaining a binary image.

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/Thresholding.webp)
*example of image segmentation: before (left) and after (right) segmentation. image source: [here](https://en.wikipedia.org/wiki/Image_segmentation#/media/File:Polarlicht_2_kmeans_16_large.png)*

**Image segmentation** is relevant in many contexts, especially in the domain of medical image analysis (identify organs, etc…). However, many algorithms are quite computationally expensive (like U-net) and thresholding has the advantages of being easy to implement and computationally efficient.

Many of the algorithms rely on the histogram of the image. If the histogram presents a [bimodal distribution](https://en.wikipedia.org/wiki/Multimodal_distribution) (with two peaks) is easier to achieve a good separation (since probably one peak represents the foreground and the second the background). When you have a bimodal histogram it is easy to separate and choose a value that allows you to separate the two peaks. Once you choose this value or threshold, you will separate all the pixels with a lower intensity value and all the pixels with a higher value. Often, the histogram is not easy to interpret and you need something more sophisticated (fear not, there are many nice available in Python that I will show you).

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/Thresholding1.webp)
*adapted from Wikipedia [here](https://en.wikipedia.org/wiki/Thresholding_(image_processing))*

Thus, when it is to identify an ideal value for separating your foreground from your background you use an algorithm to find automatically a threshold value (automatically thresholding)

### Thresholding classification

We distinguish two main categories:

* Simple thresholding, where all the pixel intensity values are selected if they are higher than the threshold value (basically we compare pixel by pixel with the thresholding value)
* Threshold computing methods where we use an algorithm to identify the best threshold value

In general, we define also **global** and **local threshold** techniques. In the **global threshold**, we use only the histogram of the image which can be affected by noise, contrast, saturation, and shadows. Therefore, a simple global threshold can lead to poor results.

For this reason, global thresholding has been helped with the use of **local properties** of images. The local methods are generally divided into histogram improvement methods and threshold computing methods.

The idea of **histogram improvement** is that we want to facilitate the separation operating on the histogram. As said before an image that presents a bimodal distribution in the histogram is easier to separate. However, these models are sensitive to noise and do not work well if we do not have a neat valley between our peaks. They are generally not very useful if we have more objects and/or complex backgrounds. Moreover, they are insensitive to small objects in the image (which is dangerous for many analyses).

Threshold computing methods have been proposed for images with complex backgrounds or multiple objects. In general, these methods are also robust to noise. The idea is that these models work with grey-level information for the selection of a suitable threshold value.

### Thresholding

Let’s go a bit more mathematically. Thresholding can be defined as a segmentation algorithm to eliminate redundant information. Simply, all the pixels with a value under a certain k (an arbitrary number) will be set to zero, while all the pixels with a value above k will be set to 255. After thresholding; we have a binary image, with only black or white pixels.

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/Thresholding2.webp)

Thresholding is often part of the pipeline for removing the background and it works better if the histogram presents a bimodal distribution (two separate peaks, making it easy to select the right k). You can try different values of k or use some specific algorithm for automatic thresholding.

If you want to test different values a wide range of variants has been proposed:
* **Arbitrary threshold**, where you choose the value you prefer (take the histogram and choose the value you prefer)
* **Mean global threshold**, you use as a threshold the mean intensity value of the image.
* **Median**

The simple threshold can be obtained with:

* **Binary threshold**, as we discussed above, you select a threshold value k, then pixels with intensity below k are set to zero while pixel with intensity above k is set at 255. You obtained after this step a binary image where everything is black or white (and looks like a Rorschach test’s ink card)
* **Threshold to zero**, the process is similar, you choose a value k and you compare the pixel intensity value. Below the k, you assign to the pixel the value 0, otherwise, you assign the value k
* **Truncated threshold**, is another variant, where you always use a value k as a threshold. The pixel with intensity above k will be set to k, the other will maintain its value.

You can implement this method very simply in Python, just using [NumPy](https://numpy.org/). For convenience, I will use the mean intensity value as a threshold.

```python 
#binary thresholding
image = im
thresh = np.mean(image)
binary = np.where(image > thresh, 255,0)
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/Thresholding3.webp)

```python 
# thresholding to zero
image = im
thresh = np.mean(image)
binary = np.where(image > thresh, thresh,0)
```
![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/Thresholding4.webp)

```python 
# truncated threshold
image = im
thresh = np.mean(image)
binary = np.where(image > thresh, thresh,image)
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/Thresholding5.webp)

### Automatic thresholding

[Otsu’s thresholding](https://en.wikipedia.org/wiki/Otsu%27s_method) is a popular method that assumes that the image contains two classes (i.e. foreground and background). Based on the histogram it calculates the threshold value k to try to minimize the combined variance of the two classes (a weighted variance for each class).

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/Thresholding6.webp)
*adapted from Wikipedia ([here](https://en.wikipedia.org/wiki/Otsu's_method))*

Considering the combined variance σ2 (for a specific value of k) is equal to the variance of each class (considering a value of k) multiplied for a certain k.

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/Thresholding7.webp)

the algorithm iteratively tries different k until the combined variance is maximized (and then this is the found k).

```python 
from skimage.filters import threshold_otsu
#otsu's thresholding

image = im
thresh = threshold_otsu(image)
binary = image > thresh
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/Thresholding8.webp)


### Adaptative thresholding

[Otsu threshold](https://en.wikipedia.org/wiki/Otsu%27s_method) provides us with only one threshold value for the image. For simple images this is not an issue, however, there are cases where this is a problem like when the light is not uniform across the image.

In these cases, we can use adaptive thresholding where we consider small neighbors of pixels and find the optimal threshold value k for each neighbor. In other words, we select a small size box of pixels and we calculate this box as the optimal threshold. This method allows handling where there are dramatic intensity changes in different parts of the images. In fact, the assumption is that smaller regions of an image are more likely to have approximately uniform illuminations. The idea is for a pixel p we select n neighbor pixel to calculate k. From this derive, an important parameter is to decide the dimension of the box around our pixel p. Ideally, this region has to be enough large to cover enough background and foreground pixels (you normally test different alternatives, this value changes for each image). Normally, k is obtained using the mean of the image box pixels Ib minus a constant C:

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/Thresholding9.webp)

Notice that the function can be different:

* Mean
* Median
* Gaussian filter
* But also custom function

In Python using the mean and after the Gaussian, we are using the skimage package:

```python 
#adaptative thresholding
from skimage.filters import threshold_otsu, threshold_local
image = im
block_size = 15
local_thresh = threshold_local(image, block_size, offset =0.01)
binary_local = image > local_thresh
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/Thresholding10.webp)

```python 
#adaptative thresholding: gaussian
from skimage.filters import threshold_otsu, threshold_local
image = im
block_size = 15
local_thresh = threshold_local(image, block_size, 
                               method= "gaussian", offset =0.01)
binary_local = image > local_thresh
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/Thresholding11.webp)

Image thresholding is one of the most commonly used techniques in [image pre-processing](https://link.springer.com/chapter/10.1007/978-1-4899-3216-7_4) tasks. It is a fast and easy alternative to image segmentation (or at least in some cases). It is often used as the basis for medical image analysis. In general, you need to test different values for the threshold.

A simple threshold works well when the image has a bimodal distribution or high contrast, but it is not adequate when the histogram separation between foreground and background is difficult. Otsu’s threshold can provide a fast and good choice for selecting a threshold value in this situation. However, when the image intensity range changes between regions of the image, the global threshold (or a single threshold value for all the images) does not work well and you can test the adaptive threshold (or local threshold).

The code can be found **[here](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/Thresholding.ipynb)**


## Neighborhood image processing

### Help thy neighbors

We have seen before point processing where we apply a function to an input image and we obtain a new image. Each pixel is modified according to the function applied: for example, a brightness increase what we increase is the intensity value of each pixel. Each pixel is modified indifferently from it is neighbors. In reality, an image is not only a pixel itself but a group of pixels: to design complex patterns we need more than a pixel.

A simple example is when we want to detect an edge, but we cannot detect an edge using the information in a single pixel. Notice in this example image, that an edge is transitioned between pixels with similar intensities to a zone where the intensities change. In neighbor processing the neighbors matter! **We modify the value using the neighbors’ pixel information.**

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood.webp)
*Image from the author (the skull radiography is coming from [here](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Lat_lebka.jpg/330px-Lat_lebka.jpg))*

To who are interested in [computer vision](https://en.wikipedia.org/wiki/Computer_vision) will notice that many of these concepts will be applied and reused. Indeed, understanding the concept as a median filter, edge detection, and Convolution would be useful in many contexts (and also when we will discuss [convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network)).

### Neighborhood preprocessing

Differently from point processing, here the output pixel value is modified according to the neighbor pixels. In this case, the pixels around a pixel contribute to its output value. Some processing techniques are used to [denoise images](https://paperswithcode.com/task/image-denoising) (as an example, some artifacts where you have some pixels with totally different values (0 or 255) in between others).

For instance, we can take the noisy pixel and use the surrounding pixels (a matrix of 3x3 pixels) we then calculate the mean and impute the [mean](https://en.wikipedia.org/wiki/Mean) (rounded to the close integer). The matrix can be larger (always odd since is centered on a pixel: 3x3, 5x5, 7x7) and can be defined also by its radius from the pixel (1 for 3x3, 2 for 5x5, and so on). As said we can choose an arbitrary number n of surrounding pixels:

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood1.webp)

As an example, the substitution of a pixel with a mean value:

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood2.webp)

Technically, to find a noise we can look for each extreme value. Generally, all the pixels in the image are substituted by the mean of the surrounding pixels. This process is called filtering and to each pixel is applied the same process. This is called the **[mean filter](https://www.sciencedirect.com/topics/computer-science/mean-filter)** when we use the mean. There are some variations:

* Local means, what we discussed above.
* In percentile mean, the filter considers only the pixel between two percentages selected by the user (p0 and p1)
* Bilateral mean

In Python this is easy to implement:

```python 
matplotlib.rcParams['font.size'] = 12

from skimage.filters import rank
from skimage.morphology import disk
selem = disk(5)

# Load an example image
img = im

percentile_result = rank.mean_percentile(img, selem=selem, p0=.1, p1=.9)
bilateral_result = rank.mean_bilateral(img, selem=selem, s0=500, s1=500)
normal_result = rank.mean(img, selem=selem)
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood3.webp)

if we use the median instead of the mean, we have a **[median filter](https://en.wikipedia.org/wiki/Median_filter)**. You can also select the **minimum** or the **maximum** value of the matrix, but these are used much less. The median filter is generally preferred, giving more accurate results.

For convention, a filter is also described by its radius, if we have a matrix of 3x3 pixels we would say the filter has a radius equal to 3. The higher (larger radius) the matrix chosen, the stronger the filter and the more computationally expensive (all the pixels are scanned by the left corner).

```python 
### MEDIAN MINIMUM MAXIMUM
matplotlib.rcParams['font.size'] = 12

from skimage.filters import rank
from skimage.morphology import disk
selem = disk(5)

# Load an example image
img = im

minimum_result = rank.minimum(img, selem=selem)
maximum_result = rank.maximum(img, selem=selem)
median_result = rank.median(img, selem=selem)
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood4.webp)

**[Salt and pepper noise](https://www.geeksforgeeks.org/difference-between-salt-noise-and-pepper-noise/)** is a method to add some random noise to an image. It mimics a real case where some disturbance of the signal. The name derives from the fact you have black and white points (pixels with value 0 are black, pixels with intensity value 255 are white). Thus, to apply salt and pepper, some pixels are randomly changed to a value of 0 or 255.

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood5.webp)
_Image source: [here](https://en.wikipedia.org/wiki/Salt-and-pepper_noise)_

Notice that these pixels are looking isolated since they have a different value from their neighbors. **What can you do in this case?**

As said before, an idea is to substitute these pixels with a different intensity value. Ideally, this value has to be similar to the neighbors, therefore the **mean filter is a good choice** (as said we are substituting the mean value of the neighbors to that pixel). **However, why do not try also what happened with the minimum and maximum filters?**

```python 
import random

def salt_pepper_noise(img, floating = True):
  row , col = img.shape
  if floating== True:
      white = 1.
      black = 0.
  else:
      white = 255
      black = 0
  n_pixels = random.randint(0,2000)
  print("n pixel modified:")
  print(n_pixels)
  for i in range(n_pixels):
    y_coord=random.randint(0, col - 1)
    x_coord=random.randint(0, row - 1)
    img[x_coord,y_coord ] = white
  for i in range(n_pixels):
    y_coord=random.randint(0, col - 1)
    x_coord=random.randint(0, row - 1)
    img[x_coord,y_coord ] = black
  return img
```
![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood6.webp)

Another point is since it considers the neighbors, the border pixels are eliminated (a pixel on the first row has no pixels above). Therefore, this is leading to reduce the image. The **[border problem](https://staff.fnwi.uva.nl/r.vandenboomgaard/ComputerVision/LectureNotes/IP/Images/ImageExtrapolation.html)** has also been considered by [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network). There are two solutions if needed: acting on the input image (duplicating the border pixels) or on the output image (duplicating the pixels after the transformation) or using a filter with a special size.

### Correlation or convolution

The method is very similar to what is seen for the median filter, however, in this case, the filter is called [kernel](https://en.wikipedia.org/wiki/Kernel_(image_processing)) and there are few differences.

**A kernel is essentially a matrix with different numbers**, we assign a position to each spot of the matrix (for the convention, indicate as h(x,y) and the center is (0,0) position). The kernel matrix is sliding from the left-top corner to the down-right corner, when we calculate the value for a pixel in the original image f(x,y) we consider it neighbors and we multiply for the corresponding value in the kernel. Once all the products, we sum them all together and this is the value in g(x,y) for the corresponding pixel. You can see this in the figure:

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood7.webp)

Mathematically, considering a kernel of radius R (i.e. 3x3 is 1) we note this as:

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood8.webp)

To avoid overflow you can normalize the kernel weight, by dividing the weight by the dimension (for example you have a 3x3 kernel you divide by 9 for each weight).

If we consider a kernel with 1 for all the positions this is a **mean kernel** (or mean filtering), that is sometimes used for blurring images (a larger kernel is blurring the image). Another often used is the **[Gaussian kernel](https://en.wikipedia.org/wiki/Gaussian_blur)** where the number normally decreases starting from the center.

**Correlation and convolution** are often used with the same meaning, while convolution is just a rotated kernel.

We are defining a function for the mean kernel and a function for the Gaussian kernel:

```python 
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return kernel

def mean_kernel(size):
    kernel =np.ones((size, size))
    kernel = kernel/size
    return kernel
```

And then apply it to the image. Notice I am also defining an arbitrary kernel of the same size:

```python 
filt1 = np.array(gaussian_kernel(5, sigma=3))
conv1 = convolve(img, filt1, mode='constant', cval=0.0)
filt2 = mean_kernel(5)
conv2 = convolve(img, filt2, mode='constant', cval=0.0)
filt3 = np.array([[0.9,0.9,0.8,0.9,0.7,0.9,0.9],
                  [0,0,0,0,0,0,0],[0,0,0,0,0,0,0]])
conv3 = convolve(img, filt3, mode='constant', cval=0.0)
```

These are the results:

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood9.webp)

# Edge detection

An important application of correlation is [edge detection](https://en.wikipedia.org/wiki/Edge_detection). An edge in an image is defined as a position where there is a significant change in the intensity value of the pixels (in a grayscale image).

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood10.webp)
*image source: [here](https://en.wikipedia.org/wiki/Edge_detection)*

Edge detection is important to define the contour of an object, measure the dimensions, and so on. Technically, we can define an edge as the pixel position when there is a significant value change (gray level).

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood20.webp)

Edge detection is based on **[gradient](https://en.wikipedia.org/wiki/Gradient)**. In a function curve, the gradient is the slope of the curve at a certain point (which is also called a tangent).

Applying this to the intensity values of the image, we can consider the edge where there are the steepest hills. The image is a 2-D, so we have two gradients (on the x, and y) and this leads to a tangent plane instead of a [tangent line](https://en.wikipedia.org/wiki/Tangent). We can calculate the gradient using the [first-order derivate](https://en.wikipedia.org/wiki/Derivative). The intersection for each point in the image is now a line (or a [vector](https://en.wikipedia.org/wiki/Vector_(mathematics_and_physics))) instead of a point:

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood12.webp)

As seen in [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), the gradient points to the steepest direction, if you want to descend the curve you have to go in the other direction. Gradient has a direction and a magnitude, we can calculate the magnitude (like how steep):

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood13.webp)

The approximation is computationally faster, and actually, we need the approximation since the first-order derivate is used for continuous curves. Images instead are [arrays](https://scikit-image.org/skimage-tutorials/lectures/00_images_are_arrays.html), so we have a quantified position, therefore we can calculate the approximate gradient for a point. The gradient is the slope between two contiguous points in a curve, here we use the difference between the values of two contiguous pixels. For a pixel in position f(x,y):

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood14.webp)

The gradient is positive when we have an increase (meaning we are encountering an edge and we are passing from a dark zone to a bright one) and negative for a decrease in values. A convenient way to do this calculation is to use a 3x3 kernel (coefficients: -1, 0, 1). Better methods and less sensitivity to noise are the [Prewitt](https://en.wikipedia.org/wiki/Prewitt_operator) and [Sobel kernels](https://en.wikipedia.org/wiki/Sobel_operator). Generally, after this filtering, you use a thresholding algorithm:

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood15.webp)

Let’s try to implement it, but notice the results are not very enthusiastic.

```python 
from skimage import filters
from skimage.util import compare_images

image = im

edge_prewitt = filters.prewitt(image)
edge_sobel = filters.sobel(image)
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood16.webp)

Just to better visualize I will show you another image.

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood17.webp)

### Canny edge detection

[Canny edge detection](https://en.wikipedia.org/wiki/Canny_edge_detector) is an algorithm that was developed by [John F. Canny](https://en.wikipedia.org/wiki/John_Canny) in 1986. It is widely used in many various computer vision systems and I will describe it here in short. The algorithm is a multi-stage process that can detect a wide range of edges. In short Canny edge detection follows 5 steps:

* **Noisy reduction**, since the gradient can be sensitive to noise in the first step you apply a Gaussian blur to smooth it (a convolution step with a Gaussian kernel)
* **Gradient calculation**, the gradient detects the edge intensity and the direction. You first convolve [Sobel kernels](https://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm) and then calculate the gradient to identify the edges.
* **Non-maximum suppression**, after the first step you will have thick and thin edges, this step is used to mitigate the thick edge
* **Double threshold**, the double threshold step selects pixels that can be considered relevant for an edge. In other words, after the first three steps, you have strong pixels (high intensity) and weak pixels (low intensity but still in the acceptable range), and then you have non-relevant pixels (basically noise). In this step we apply a threshold to identify strong pixels, a threshold to filter out the non-relevant, and what is in the middle is considered a weak pixel.
* **Edge tracking by hysteresis**, the hysteria mechanism transforms a weak pixel into a strong pixel if the weak pixel has a strong pixel as a neighbor

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood18.webp)
_adapted from [here](https://en.wikipedia.org/wiki/Canny_edge_detector)_

Well, at this point let’s try it out. [Skimage](https://scikit-image.org/) provides a very good implementation where you can control the internal parameters:
* **Sigma** is controlling the standard deviation of the Gaussian filter.
* **high_threshold**, it is controlling the upper threshold to identify the strong pixels
* **low_threshold** controls the threshold for filtering out the non-relevant pixels.

Check also the other optional parameters [here](https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.canny). Let’s try the different values of Sigma to check the results.

```python 
from skimage import feature
img =copy.deepcopy(im)
# Load an example image


im1 = feature.canny(img)
im2 = feature.canny(img, sigma=1.5)
im3 = feature.canny(img, sigma=3)
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/neighboorhood19.webp)

Much better! Notice also how the different values of sigma are influencing the results.

We have seen different image transformations that take into account also the value of the neighbors, thus here an important hyperparameter is the number of neighbors we are considering. These transformations are useful in many fields and they are the basis of many complex and sophisticated algorithms in computer vision. Convolution, for example, is the basis for the convolutional neural network which has revolutionized computer vision allowing it to solve complex tasks such as [image classification](https://paperswithcode.com/task/image-classification), [pattern recognition](https://en.wikipedia.org/wiki/Pattern_recognition), and [segmentation](https://www.ibm.com/think/topics/image-segmentation). Edge detection is often a fundamental step in image analysis and if you think about humans are naturally doing that (when you see something and then you are sketching on a paper, you are most likely drawing the perceived edges).

To be concise here I show the essential code, **but all the codes used are present [here](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/neighborhood_processing.ipynb)**

## Morphological image processing

### Introduction to morphology

[Morphology](https://en.wikipedia.org/wiki/Mathematical_morphology) (sometimes referred also as mathematical morphology) can be considered a branch of neighborhood processing. It was developed in 1964 by [Georges Matheron](https://en.wikipedia.org/wiki/Georges_Matheron) and [Jean Serra](https://en.wikipedia.org/wiki/Jean_Serra) to quantify characteristics of mineral cross-sections but proved to be valuable in many other applications. In general, morphology can be used to remove the noise originated by a first thresholding step (which often happens in images where the exposition is not uniform). In fact, morphology works very well with the binary images obtained by thresholding (but you can also use it on grayscale images).

Just as an example, a few cases can happen after thresholding and can be solved by morphology.

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology.webp)

Here I have applied hit or fit operations, dilation, and erosion, which will be discussed below.

Morphology has different interesting applications, for instance, is used as a preprocessing step in [optical character recognition (OCR)](https://en.wikipedia.org/wiki/Optical_character_recognition), detecting barcodes and license plates. Morphology operations are simple and computation not expensive and can be combined together, thus an efficient use of morphology can save time and computation resources. Indeed, often you do not need a complex algorithm to perform different tasks, less advanced techniques can lead to an elegant and efficient solution. Moreover, these operations are very useful in different computer vision algorithms and indeed they are worth learning.

### Morphology operations

Thresholding is a global operation (on all the pixels in the image) without considering the local position, which can lead to under/over-segmented regions.

Instead, morphology is applied similarly to neighborhood processing, in this case, a kernel with only 0 and 1 is applied (the kernel is also called the structuring element). In this case, what is important is not the values but the shape (the disposition of the 1, where the box-shaped kernel preserves sharp corners and round/disc kernel rounds the corner)

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology2.webp)

The kernel is applied only to the objects (or elements) present in the image. Therefore, the bigger the kernel bigger the effect on the image. Often these procedures are done in the binary images (like generated by thresholding, where 1 means foreground and 0 background).

**Hit and fit**

[In this algorithm](https://en.wikipedia.org/wiki/Hit-or-miss_transform), we start putting a kernel like the one we have seen above in a certain position and then we consider the value of the pixels around that are covered by the kernel. The idea is to understand if considering a kernel, the pixel in a certain position is 1 when is also 1 in the kernel (this is called hit). If there is a concordance, the pixel is set to 1 in the output image. Noticing that in hit (but also in fit) we are not scanning the whole image, we are just selecting a position and our kernel and checking the concordance of our kernel and the pixels in the position.

In the case of fit, we check if all the pixels at the same position are 1 as it is in the kernel (if it is true, the image is fitting). If there is concordance for all pixels, all the pixels are set to 1 in the output image, otherwise 0 for all. Applying hit or fit on the two-position below:

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology3.webp)

**Dilation and erosion**

The application of hit on the entire image is called **[dilation](https://en.wikipedia.org/wiki/Dilation_(morphology))** because the elements in the image are size-increased after the transformation. Moreover, small holes are closed and some objects are merged. The increase depends on the size of the kernel element or in the alternative, apply a small kernel iteratively. The problem is that also noisy objects will be enlarged. The equation is with kernel k:

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology4.webp)

⊕ is representing the sum of vectorial sub-spaces.

As said, remember that the size of the kernel impacts the effect and the effect is similar when applying iteratively a smaller kernel (ex 6x6 kernel has a similar effect of 2 times a 3x3 kernel). Let’s try the effect of different kernels on an image. We will start with an image where we apply Otsu’s thresholding (as said before in a precedent tutorial, this returns a binary image where the pixels over a certain threshold have 255 or white and the rest is zero).

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology5.webp)

Then we apply this binary image to different kernel sizes and dilatation:

```python 
#dilatation 
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import disk
from skimage.morphology import erosion

fig, axes = plt.subplots(ncols=4, nrows = 1, sharex=True, sharey=True,
                         figsize=(12, 5))

image = im
thresh = threshold_otsu(image)
binary = image > thresh

dilated = ndimage.binary_dilation(binary, structure=np.ones((3,3)))
dilated1 = ndimage.binary_dilation(binary, structure=np.ones((5,5)))
dilated2 = ndimage.binary_dilation(binary, structure=np.ones((9,9)))
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology6.webp)

Notice that the object is becoming bigger, the holes are filled in the image and also some noisy elements are enlarged.

**[Erosion](https://en.wikipedia.org/wiki/Erosion_(morphology))** is contrary to dilatation, in this case, we are applying fit to all the images. The effect is a general reduction of the size of the objects with the elimination of small objects. Moreover, larger objects are often split into smaller objects. We are eliminating noise but objects of interest become fractured.

Here is the equation:

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology7.webp)

here is the Python implementation:

```python 
#erosion

fig, axes = plt.subplots(ncols=4, nrows = 1, sharex=True, sharey=True,
                         figsize=(12, 5))

image = im
thresh = threshold_otsu(image)
binary = image > thresh

eroded = ndimage.binary_erosion(binary, structure=np.ones((3,3)))
eroded1 = ndimage.binary_erosion(binary, structure=np.ones((5,5)))
eroded2 = ndimage.binary_erosion(binary, structure=np.ones((9,9)))
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology8.webp)

Notice, that small objects are disappearing, also there forming holes in the objects of interest,

From the combination of erosion and dilation, we derive compound operations like opening, closing, and boundary detection.

**Closing and opening**

[Closing](https://en.wikipedia.org/wiki/Closing_(morphology)) is generally the operation to close holes; it is obtained by dilatation followed by erosion. The internal holes in the image are normally closed after this operation. Using dilation, we are increasing the size of the object (and of the noise), since the output object has the same input size, closing is solving this problem. The kernel for the subsequent operation has the same size. The closing operation is idempotent, meaning you can use it only once otherwise you just shrink the whole image without a noticeable effect (the border problem). the equation:

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology9.webp)

Notice here what is happening when we are using a different size of the kernel:

```python 
closed = ndimage.binary_closing(binary, structure=np.ones((3,3)))
closed1 = ndimage.binary_closing(binary, structure=np.ones((5,5)))
closed2 = ndimage.binary_closing(binary, structure=np.ones((9,9)))
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology10.webp)

[Opening](https://en.wikipedia.org/wiki/Opening_(morphology)) is generally used to avoid fractioning bigger objects when removing the noise. In this case, we use first erosion and then dilation. The output image presents an object with the original size but the noise is removed. Another [idempotent transformation](https://en.wikipedia.org/wiki/Idempotent_matrix) and the equation is:

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology11.webp)

Let’s also test the opening operation to see what is happening:

```python 
opened = ndimage.binary_opening(binary, structure=np.ones((3,3)))
opened1 = ndimage.binary_opening(binary, structure=np.ones((5,5)))
opened2 = ndimage.binary_opening(binary, structure=np.ones((9,9)))
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology12.webp)

You can also combine the two processes, but the kernels have to be different (the one for opening and the one for closing)

Let’s give a look at all the operations together using a 5x5 kernel.

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology13.webp)

**[Boundary detection](https://paperswithcode.com/task/boundary-detection)** is an edge detection technique on [binary images](https://en.wikipedia.org/wiki/Binary_image), where you subtract the eroded image, obtaining the boundary. The idea is that with eroding we are obtaining a smaller version of the object and if we subtract the image only the boundary will remain. In formula:

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology14.webp)

```python 
eroded = ndimage.binary_erosion(binary, structure=np.ones((3,3)))
boundary =binary ^ eroded
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology15.webp)

Notice that we are not subtracting but since we have two logical masks (true/false) we are using the logical operator AND otherwise, [Numpy](https://numpy.org/) is returning an error (but the principle is the same)

### A little practical example

Let’s say we want to select all the nucleus contours in a microscope image to do some analysis, with just a few operations:

```python 
#we want to select only the nucleus
#thus only what is blue
im = a[:,:,2]
image = im
thresh = threshold_otsu(image)
binary = image > thresh
eroded = ndimage.binary_erosion(binary, structure=np.ones((7,7)))
opening = ndimage.binary_opening(eroded, structure=np.ones((11,11)))
boundary =binary ^ opening
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology16.webp)
*original image from Wikipedia commons*

Bonus example, if you want to read the license plate on a car image, instead of a complex deep learning model you can start with simple pre-processing steps. You can use some simple operations like a white and black hat (white subtracts the opening image from the gray image, while the black hat subtracts the closing from the gray input image). Notice here we are using a rectangular kernel because the plate is wider than taller and we can use a kernel of arbitrary size. Just a few simple operations and results are quite nice:

```python 
image = im
thresh = threshold_otsu(image)
binary = image > thresh
opening = ndimage.binary_opening(binary, structure=np.ones((13,5)))
closing = ndimage.binary_closing(binary, structure=np.ones((13,5)))
black_hat =  im - closing
white_hat = im - opening
```
![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology17.webp)
*original image from [Łukasz Nieścioruk](https://unsplash.com/@luki90pl) at unsplash.com. On the left, is the image after applying morphological operations (image by the author).*

We have seen how powerful are morphology operations, where with simple operations we can obtain different results. It is worth noticing that each operation has its own contrary and you can combine them together to complete more sophisticated tasks.

In synthesis, you can use the technique of erosion to remove small links that are connecting your objects, remove small noise objects but also subtract to the binary image detect boundaries. Dilation on the other side is useful to connect parts of images. The opening allows you to remove small objects without fracturing and decreasing their objects, while the closing allows filling holes without increasing the object size. And you can combine these operations in other iterative and clever ways, according to your needs. **Not bad for simple operations, right?**

To be concise here I showed the essential code, but all the codes used are present **[here](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/Morphology.ipynb)**

## A Practical Guide to BLOB Analysis and Extraction

### What is a Blob?

If you search Google for “blobs”, you can find these astonishing images. By the way, this is [P. polycephalum](https://en.wikipedia.org/wiki/Physarum_polycephalum), an acellular slime mold commonly as the blob. While being a mold, researchers found that can find the solution for the shortest path problem (basically, the blob could pass a FAANG interview since a recurring question is about the shortest path algorithm).

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/blob.webp)
*P. polycephalum, aka the blob: Image source from [CNRS phototeque](https://images.cnrs.fr/)*

**It looks a bit of a digression, but actually, what do we intend for a blob in image analysis?**

An example is when we have some cells in a microscope image and we want to count them. Notice, that we have many small circles and we want to know how many there are, or better let the computer do this job. Clearly, the first step is to separate them into individual entities and only after counting them.

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/morphology16.webp)
*original image from Wikipedia commons*

The idea is to identify objects (a group of connected pixels over a certain size) and to analyze the objects. Generally, the BLOBs are binary images, but certain algorithms also work with gray-level images. Moreover, in some cases, you do not want just a number, but you want more information. Like you have a microscope image of blood cells, some cells are red blood cells, and others are white blood cells, and you want a single number for each category. These two examples represent blob extraction and blob classification.

### Blob Detection and Extraction

In a general definition, we can say that the idea of **[BLOB detection](https://en.wikipedia.org/wiki/Blob_detection)** is to detect blobs in the image, and **[BLOB extraction](https://en.wikipedia.org/wiki/Connected-component_labeling)** is to separate the blob objects in the image. A BLOB is basically a group of connected pixels in an image that share some common property (like the intensity value).

A **[BLOB algorithm](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html)** normally returns a labeled array, each separate object gets a label (normally a number) and each pixel of the object gets the same label. To define if two pixels are part of the same object we have to see if they are connected. In fact, many of BLOB’s extraction algorithms are called **[connected component analysis](https://en.wikipedia.org/wiki/Connected-component_labeling)**. There are two types of connectivity 4-connectivity (the four directions: right, left, up, down) and 8-connectivity (diagonal corner included). You can implement them with different kernels. The latter is generally more accurate but computationally more expensive.

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/blob2.webp)

Notice what is happening in two different cases if you are using the different types of kernels

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/blob3.webp)

### The Grass fire algorithm

**[The grass-fire algorithm](https://en.wikipedia.org/wiki/Grassfire_transform)** is one of the most used and it scans the whole image looking for objects starting from the upper-left corner. When it finds a non-zero (an object pixel in a binary image), it checks in the neighborhood (the four directions in 4-connectivity). A not-zero pixel if is attached is considered connected, if the algorithm encounters a zero pixel it stops the search in that direction. When it cannot extend the search for the object, it considers that object separated from other objects. All the pixels of one object share the same label. Then it starts the search again until it gets the next object pixel.

Let’s try the grass fire algorithm on a histology image.

```python 
from skimage.filters import threshold_otsu
from scipy import ndimage
image = im
thresh = threshold_otsu(image)
binary = image > thresh
eroded = ndimage.binary_erosion(binary, structure=np.ones((12,12)))
from skimage import measure
all_labels = measure.label(eroded)
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/blob4.webp)
_original image from [scikit-image](https://scikit-image.org/) (left), then adapted and modified by the author_

The results look interesting, we could detect and separate different parts of the image. This could be useful for further analysis.

There are also other algorithms that can be used:

* **[Laplacian of gaussian (LOG)](https://automaticaddison.com/how-the-laplacian-of-gaussian-filter-works/)**. The most accurate but also the slowest approach, detecting bright blobs on dark backgrounds. It works by computing the Laplacian of Gaussian images, increasing progressively the standard deviation and stacking one over the other as if it were a cube. The local maxima in this cube are the blobs.
* **[Difference of Gaussian (DoG)](https://en.wikipedia.org/wiki/Difference_of_Gaussians)**. It is a fast approximation of the LOG approach, but it also assumes that blobs are bright objects on a dark background. It works by blurring images with increasing standard deviation.
* **Determinant of Hessian (DoH)**. The fastest approach and do not assume blobs are only bright on dark (but they can be also dark on bright). However, while it works better with large blobs it detects less accurately smaller objects.

Let’s give them a try and see which one performs better. Well, will use this image of nuts to check our three algorithms.

Why? Because they are round-shaped but still with a complex shape and different shadows allowing us to test better our algorithm (and also because personally, I think that nuts remember the blobs).

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/blob5.webp)
_image source from [Priyanka Singh](https://unsplash.com/it/@priyankasingh) at [unsplash.com](https://unsplash.com/it)_

In Python, we need a few lines of code to test it:

```python 
from skimage.feature import blob_dog, blob_log, blob_doh
blobs_log = blob_log(closed, max_sigma=30, num_sigma=10, threshold=.1)
blobs_dog = blob_dog(closed, max_sigma=30, threshold=.1)
blobs_doh = blob_doh(closed, max_sigma=30, threshold=.01)
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/blob6.webp)

Notice that Laplacian of Gaussian and Difference of Gaussian performed better than a determinant of Hessian. The first method is also finding some small blobs for each nut.

### BLOB Features

The aim is to extract **relevant features** from the BLOB, which can be in turn used to classify them. In other words, the idea is to reduce the blobs to some summary values (which can be fed to a [classifier](https://www.datacamp.com/blog/classification-machine-learning)). For example, you can use these simple features for many tasks, like when you are selecting cells, you can exclude no round blobs. Normally, we exclude the BLOB that is in contact with the border (since we do not know if it is a complete BLOB or just part of something). We can extract:

* **Area** or number of pixels composing the BLOB. the pixels for each labeled BLOB are counted. Often the area is a criterion to remove objects too small or too large.
* **[Bounding box](http://d2l.ai/chapter_computer-vision/bounding-box.html)**. It is a rectangle containing the BLOB (the minimum possible) referring to x and y as the position of the BLOB:
![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/blob7.webp)
* The **bounding box ratio**, is used to calculate the elongation of the box (obtained by dividing the height by the width)
* **Bounding circle**, sometimes you are interested in a circle. Found the center of the BLOB, you look in all directions until you find the fairest BLOB point, the line connecting these two points is the radius of the bounding circle.
* **Convex hull**, which is a polygonal shape box containing our BLOB. starting from the top, as it is a rubber band we are looking for a new extremity. From the new point, it starts another search until the polygon is complete.
* **Compactness**, in simple words, is the ratio between the BLOB area and the bounding box (and it can be useful for differentiating objects of the same dimension but different shapes).
* The **Center of mass or centroid**, for a binary image, is the average x and y position. The calculation for the coordinates of this point is done by the [arithmetic mean](https://en.wikipedia.org/wiki/Arithmetic_mean): for the x coordinates we sum all the x positions of all the points and then divide by the number of pixels (for y coordinates the same). More formally:
![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/blob8.webp)
* The **perimeter** is the perimeter of the BLOB. it is the sum of the pixels of the contour of the BLOB (sometimes done using first-edge detection or other contour methods)
* **Circularity**, there are different methods for calculating circularity, Heywood’s circularity factor is calculated to consider a perfect circle with a value of 1 and a straight line as ∞ a value:
![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/blob9.webp)
* **Inverse Circularity** is often used since the value is in the range 0–1, with 1 a perfect circle and 0 a line:
![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/blob10.webp)

In Python, it is quite simple, we just need to select and separate our BLOBs, and then are straightforward to extract properties (which we can easily store in a data frame).

```python 
thresh = threshold_otsu(im)
binary = im > thresh

blobs = measure.label(binary > 0)

properties =['area','bbox','convex_area','bbox_area',
             'major_axis_length', 'minor_axis_length',
             'eccentricity']
df = pd.DataFrame(regionprops_table(blobs, properties = properties))
df
```
and these are the results:
![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/blob11.webp)

### BLOB classification

Extracted BLOB features can be used to train a classifier. Another approach used is to calculate the distance from a prototype (which is a model of what you are looking for). For example, if you are looking for circular objects you compare a perfect circle as a prototype. You can calculate the Euclidean distance from the prototype for each feature in the feature space:

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/blob12.webp)

Before extracting the features, you may need to preprocess (binarization, morphology transformation) and features can need normalization. Then you can train your model and evaluate it.

### Hunting for panda
Let’s see a little example, we have these two beautiful pandas and we want to isolate them in the image.

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/blob13.webp)
_image source from [Pascal Müller](https://unsplash.com/it/@millerthachiller) at [unsplash.com](https://unsplash.com/it)_

We then load the image in Python and we use thresholding. We have used Otsu’s thresholding to find the optimal threshold, however, there are so many non-panda pixels that passed the threshold. We care about our sad panda cubs and thus we are also open to reducing the background.

```python 
im = color.rgb2gray(a)

image = im
thresh = threshold_otsu(image)
binary = image > thresh
opening = ndimage.binary_opening(binary, structure=np.ones((5,5)))

plt.imshow(opening, cmap= "gray")
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/blob14.webp)

There are still too many objects, meaning we cannot separate our pandas only on the pixel intensity value. In some cases, the Otsu’s threshold does not provide a suitable mask: We are trying to extract all the blobs.

```python 
blobs = measure.label(opening > 0)
plt.imshow(blobs, cmap = 'tab10')
plt.axis('off')
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/blob15.webp)

Moreover, we can calculate the different properties of these blobs.

```python 
properties =['area','bbox','convex_area','bbox_area',
             'major_axis_length', 'minor_axis_length',
             'eccentricity']
df = pd.DataFrame(regionprops_table(blobs, properties = properties))
df.sort_values('area', ascending= False)
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/blob16.webp)

From the image, it is clear that the two biggest blobs are the panda, but sometimes this is not that clear. We could for instance use our panda knowledge and a feature that considers the ratio of major versus minor axis length (pandas are notorious for round shape).

We then select the first two blobs and voilà!

```python 
b = np.where((blobs == 24)|(blobs == 23), blobs, 0)
plt.imshow(b)
```

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/blob17.webp)

Blob extraction is a powerful pre-processing step, easy and fast to compute but also provides a lot of informative features. With a few lines of code, we could isolate objects and decide on criteria for automatic selection.

With a bit of practice, a good data scientist can use morphology operation and blob extraction to create a powerful image analysis pipeline.

To be concise here I showed the essential code, but all the codes used are present [here](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/BLOB_extraction.ipynb)

## Harnessing the power of colors

### What is a color?

The definition of [color](https://en.wikipedia.org/wiki/Color) is not an easy task. We can say that the [visible light](https://en.wikipedia.org/wiki/Visible_spectrum) hits something and absorbs some [wavelength](https://python.plainenglish.io/harnessing-the-power-of-colors-in-python-92bf6fe175c9) and then what is not absorbed is [reflected](https://en.wikipedia.org/wiki/Reflection_(physics)). Notice, that [visible light](https://science.nasa.gov/ems/09_visiblelight/) is actually a very narrow range of frequencies/wavelengths which is called the visible spectrum and it ranges from 380 to 750 nanometers. This tiny range of frequencies is actually special since it is something that a typical human eye can capture. The magic happens when our eyes collect the reflected wavelengths of an object and this light excites the human [photoreceptors](https://en.wikipedia.org/wiki/Photoreceptor_cell).

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/colors.webp)
_image source: [here](https://en.wikipedia.org/wiki/Visible_spectrum)_

**In the [human eye](https://en.wikipedia.org/wiki/Human_eye)**, the cone is specialized nerve cells that act as photoreceptors. There are three types of [cone cells](https://en.wikipedia.org/wiki/Cone_cell), and each type is sensitive to a different wavelength. There are also [rods](https://en.wikipedia.org/wiki/Rod_cell), which are more sensitive to light and support vision at a low-light level. But cones are perceiving the colors and they are much better at perceiving details. The color perception changes from individual to individual, since the percentage of the cones, can change and there are different genetic mutations (thus, we all perceive color differently).

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/colors2.webp)
_image source: [here](https://en.wikipedia.org/wiki/Cone_cell)_

As a curiosity, human eyes perceive red as being 2.6 times as bright as blue and green, that’s why a single red dot can capture our eye at first glance (yes, like a cat when you use a laser).

![example of image segmentation: before (left) and after (right) segmentation. ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/colors3.webp)
_image source: [here](https://en.wikipedia.org/wiki/Piet_Mondrian)_

### Color images

In a color image, we have three values for each pixel: one for the Red, Blue, and Green (or [RGB](https://en.wikipedia.org/wiki/RGB_color_model)). We can represent each pixel as a 3-dimensional vector, and therefore an image as a three-different array. Normally each color is encoded with [8-bit](https://en.wikipedia.org/wiki/8-bit_color) (meaning 256 shades) and a pixel could represent more than 16 million colors. We can image a pixel as a point in a 3d space, with RGB color axes starting from black (0,0,0).

Notice, that by loading an image in Python we can easily slice it in the color components:

```python 
#slicing channels 
im1 = im[:,:,0]
im2 = im[:,:,1]
im3 = im[:,:,2]
```

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/colors4.webp)
_original image from [Mark Harpur](https://unsplash.com/it/@luckybeanz) at [Unsplash](https://unsplash.com/it). Modified version by the author._

The line connecting black (0,0,0) to white (255, 255, 255) is called gray-vector. Theoretically, all the colors on the line from black (0,0,0) to red (255, 0, 0) are the same color but just different shades (different levels of illumination). This can be generalized also for the other colors, and we can define the chromaticity plane where the colors on the edge are defined as pure and moving to the center becoming gray (or polluted by the light). This concept is used to normalize the RGB color or to transform it into a large format (color plus intensity).

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/colors5.webp)

### Dramatic black and white

A color image can be **converted in [grayscale](https://en.wikipedia.org/wiki/Grayscale)**, the transformation is not invertible. The conversion is done by multiplying each channel for weight and summing up:

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/colors6.webp)

The weights have a value of 1/3 or in some contexts (for analysis purposes, if you are more interested in plant classification the green is more important) they can have different values. In visualization often, these weights are used:

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/colors7.webp)

The threshold on color images is more complex since you need to consider for each color a minimum and maximum value (a range). Since color changes with the intensity of light, this can also be problematic. Often the image is converted to zero before the threshold.

```python 
im = io.imread(url)
im1 = (0.2125 * im[:, :, 0]) + (0.7154 * im[:, :, 1]) + (0.0721* im[:, :, 2])
im2 = (0.7 * im[:, :, 0]) + (0.2 * im[:, :, 1]) + (0.1* im[:, :, 2])
im3 = (0.2 * im[:, :, 0]) + (0.2 * im[:, :, 1]) + (0.7* im[:, :, 2])
```
![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/colors8.webp)
_image source from Wikipedia._

### Counting color in an image

Many times, I was asked to match the color template when preparing a PowerPoint. The eternal struggle in trying to understand if the colors are similar. How we can know which are the predominant colors inside an image? It looks like a simple question, but using Pantone’s master book is not a feasible solution.

As you may know, RGB colors have assigned a hexadecimal code that is used to display on web pages. The hex color code is also recognized widely by Python libraries, so the idea is to obtain a list of the most represented hex colors and their count. The idea is to use a clustering algorithm to find color clusters and we transform the cluster center (centroid) into hexadecimal colors. In this way, since we are forming clusters of color we will extract the most prominent [colors](https://en.wikipedia.org/wiki/Web_colors). Then we will plot using a [bar plot](https://www.data-to-viz.com/graph/barplot.html).

```python 
def extract_colors(image, n_colors, resize = 0.5):
    """
    count colors in an image
    """
    def RGB2HEX(color):
      "RGB color to HEX colors"
      return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
    
    #scaling if too big
    if resize is not None:
      "note: in the next version  multichannel argument is changed"
      image = rescale(image, resize, anti_aliasing = True, multichannel= True)
      image = image *255

    image = image.reshape(image.shape[0]*image.shape[1], 3)
    #clustering step
    clf = KMeans(n_clusters = n_colors)
    labels = clf.fit_predict(image)
    #counting step
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))
    #obtaining HEX colors
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    #save in a pdf
    df = pd.DataFrame(columns= ["colors", "value_count"])
    df["colors"], df["value_count"] = hex_colors, counts.values()
    return df
```

Let’s start with this aerial picture

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/colors9.webp)
_original image from [Mark Harpur](https://unsplash.com/it/@luckybeanz) at [Unsplash](https://unsplash.com/it)._

We run our function, which is basically returning a data frame:

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/colors10.webp)

Then we can use [seaborn](https://seaborn.pydata.org/) to plot the bar plot:

```python 
ax = sns.barplot(x="colors", y="value_count", data=df, palette = df["colors"])
plt.xticks(rotation = 45)
```

bar plot representing the most represented colors in the image. Each column is a color:

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/colors11.webp)

Notice what’s happening using different pictures:

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/colors12.webp)
_Bar plots of the picture’s extracted colors. original images: [Mark Harpur](https://unsplash.com/it/@luckybeanz) at [Unsplash](https://unsplash.com/it) (left), from [Tijs van Leur](https://unsplash.com/it/@tijsvl) at Unsplash (middle), and from [Cassie Matias](https://unsplash.com/it/@cass4504) at Unsplash (right). Bar plots generated by the author._

Another application is the use of [k-means](https://en.wikipedia.org/wiki/K-means_clustering) to segment the image. In fact, using [clusters](https://en.wikipedia.org/wiki/Cluster_analysis) we can divide the image into segments. This approach also has the advantage that we can quickly do tests by changing the number of clusters

```python 
def segment_image_kmeans(im = None, K =3):
  '''
  segment an image with k-means
  '''
  # converting to HSV space
  img_hsv=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
  vectorized = np.float32(img_hsv.reshape((-1,3)))
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
  K = 3
  attempts=10
  ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
  center = np.uint8(center)
  res = center[label.flatten()]
  result_image = res.reshape((img_hsv.shape))
  return img_hsv, result_image

  
k=3
img_hsv, result_image = segment_image_kmeans(im = img, K =K)
```

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/colors13.webp)
_image source for the code: [here](https://unsplash.com/fr/photos/photographie-de-larene-du-colisee-VFRTXGw1VjU)_

### Coloring black and white images

An interesting application is to make images that are black and white into color. Especially for old photographs, thanks to deep learning today it is possible to recolor images

In future articles, we will discuss convolutional neural networks and other models. For now, suffice it for us to know that there is an artificial intelligence model behind this process that has been trained to color images. The code is in the notebook.

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/colors14.webp)
_original image from [here](https://unsplash.com/fr/photos/new-york-une-scene-de-harlem-names-parks-z4urTwO9nRc)_

There is a lot of information hidden in the colors that can be used in many downstream tasks. For instance, extracting colors can be used to search images for their similarity in the color footprint. We can also couple this information with shapes and other features we can extract from the images.

A more efficient way can be extracting features that consider colors, shape, and texture. As an example, we could use a pre-trained model to build a feature extractor and based on this calculate the similarity between images. In other words, as you can obtain text embedding you can do the same with images.

To be concise here I showed the essential code, but all the codes used are present **[here](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/Colors.ipynb)**

## Image Segmentation

[Segmentation](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://en.wikipedia.org/wiki/Image_segmentation&ved=2ahUKEwit36KIgp6LAxVcU6QEHftBK_wQFnoECBAQAQ&usg=AOvVaw2L4Ili5rq-sxwyCn1Mi3_q) is one of the most widely used techniques in computer vision. It can be used for a variety of applications from medical imaging to security. For example, segmenting organ images allows potential diseases to be better studied. Artificial intelligence methods are used today that require very complex and computationally heavy methods. This is not always necessary; simple methods can also be used with Python in some cases.

In this section we will discuss:

* segmentation, how to segment without the use of deep learning
* what is pixel classification, how to do and why is related to image segmentation
* what are and how to use parametric models
* How to use naive bayes and gaussian mixture models for image segmentation

[Classification](https://www.datacamp.com/blog/classification-machine-learning) is an important task in image analysis and can be used to solve different challenges. There are different examples of classification when applied to images, in this article, I will focus on the less-discussed one-pixel classification.

In this context, we want to classify the pixels to which class they belong. This task can be seen as a sort of preprocessing step where we assign each pixel a label. The methods described here assign a label to the pixels according to their values. In this case, I will not use any [Convolutional Neural Network](https://www.ibm.com/think/topics/convolutional-neural-networks), [Vision Transformer](https://paperswithcode.com/method/vision-transformer), or other heavy and complex methods. Instead, I want to show how you can obtain nice results using even simple methods.

The **[image segmentation task](https://en.wikipedia.org/wiki/Image_segmentation)** is also called semantic segmentation. Moreover, an image with pixel regions with different labels is defined as annotated (normally done by an expert). These annotated images can feed to an algorithm to annotate unlabeled images ([supervised annotation](https://en.wikipedia.org/wiki/Supervised_learning)). There are also [unsupervised methods](https://en.wikipedia.org/wiki/Unsupervised_learning) that do not require expert annotation. Here is an example of the original image after the classification of the pixel.

This task is very useful also in different fields like [medical image analysis](https://en.wikipedia.org/wiki/Medical_image_computing). These labeled regions can represent different tissues, and you can learn features, do statistics, and so on.

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/segmentation.webp)
_image source: [here_](https://en.wikipedia.org/wiki/Image_segmentation)

Notice, that you are not limited to 2D images but you can segment also volumes. In medicine, this can be useful for locating tumors, studying anatomical structures, surgery planning, virtual surgery, and measuring tissues, and anomalies. Furthermore, this technique can be useful for an autonomous car in detecting pedestrians or brake lights. It is used also to locate objects in satellite images, fingerprint and iris recognition, traffic control systems, and so on.

### Pixel classification

In classification, the aim is to find different classes in the image (a priori decided or unsupervised) and then assign pixels to these classes. More formally, Pixel classification is a subbranch of image classification, in this case, we classify pixels just according to their value (value to label) and it is often an initial step. As a convention, we consider a pixel with value v ∈ ℝ that is the class c of the class set C containing k classes. Therefore, a classification rule can be written as:

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/segmentation2.webp)

In simple words, we are defining some intervals where the value of a pixel belongs to a class. However, we still have two questions:

* How do we define that a pixel belongs to a class?
* How many classes?

The simplest method is to do it manually (however is error-prone and time-consuming). As an example, we assume to have three different classes and we define three thresholds:

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/segmentation3.webp)

We use these thresholds to assign the pixel to each class:

```python 
img =np.where(img <=threshold[0], 0, img )
img =np.where((img >threshold[0]) & (img <=threshold[1]), 1, img )
img =np.where(img >=threshold[2], 2, img )
```

Here are the results:

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/segmentation4.webp)
_Original image from Wikipedia Commons ([here](https://upload.wikimedia.org/wikipedia/commons/7/75/MeihuaShan_1.jpg)), processed image from the author (left)_

Ok, it looks like a contemporary masterpiece, something you can find in the Centre Pompidou or in the Getty Museum, but overall, our classification results look poor.

A **[minimum distance classifier](https://seos-project.eu/classification/classification-c04-p01.html)** is a simple method that assigns a pixel to a class, it calculates the distance from the pixel value to the mean µ of each class c and assigns the pixel to the closer class (i.e. if we have two classes, c1 (with range [10–40] and µ = 25) and c2 (range [60–80], µ = 70) a pixel with value v=55 have distance 30 from c1 and 15 from c2, so it will be assigned to c2). More formally:

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/segmentation5.webp)

### The parametric approach or the Gaussian triumph

**[Parametric approaches](https://www.geeksforgeeks.org/difference-between-parametric-and-non-parametric-methods/)** take into account the histogram and the class distribution (classes have different mean and variance). In general, we expect our class to have a [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) with a probability density function of:

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/segmentation6.webp)

And taking into account all the pixels of the class we can estimate the mean and the variance:

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/segmentation7.webp)

For each class, we can compute the parametric description, for a class (with mean µ ad variance σ):

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/segmentation8.webp)

The equation below is for a single class, and for another class is the same except for mean µ and variance σ. If we want to assign a pixel we substitute in each class equation v with the pixel value, then we attribute the pixel to the class where the resulting value is higher (ex we have two classes one with f1 and the other with f2, for the pixel p, we do the substitution if f1>f2, then the pixel is assigned to class 1).

However, this method is more computationally expensive (different equation to compute), you can use the equation below to compute where to set the boundary for the adjacent class. Ex. You have two classes f1 and f2, and you want to find the value v that separates the two classes, you compare f1= f2 and you solve to v. Since it is a second-degree polynomial equation, you have two solutions:

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/segmentation9.webp)

herefore, you compute the boundary of the classes, and you assign the pixel considering only the class ranges.

An example of separation using the **[Gaussian mixture model](https://en.wikipedia.org/wiki/Mixture_model)**. The idea is simple we are looking at the image and we want to separate the cells from the rest. The image we are using is [immunohistochemistry (IHC)](https://en.wikipedia.org/wiki/Immunohistochemistry) staining of the colon (from [Scikit-image data](https://scikit-image.org/docs/stable/api/skimage.data.html)). The nuclei are colored in faint blue while the brown represents the expression of a protein FHL2 (it is basically highlighting the [cytoplasm](https://en.wikipedia.org/wiki/Cytoplasm), the part of the cells around the [nucleus](https://en.wikipedia.org/wiki/Cell_nucleus)). IHC staining is routinely used in medicine by pathologists to understand if the cells are abnormal or if there is a cancerous lesion. If we want to create an algorithm that identifies cancer cells, separating them from the background is a good starting point.

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/segmentation10.webp)
_IHC staining from [here](https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_ihc_color_separation.html#sphx-glr-auto-examples-color-exposure-plot-ihc-color-separation-py)_

The process here is similar to clustering: we have inspected the image and we want to separate the cells from the background, thus we have two classes. We use the [GMM model](https://www.geeksforgeeks.org/gaussian-mixture-model/) from scikit-learn and we decide the number of classes, we reshape our image in the format [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) likes, and we train. The means_ attribute basically provides the mean for each mixture component (i.e. the Gaussian curve for each class). We will use that as a threshold and we obtain our binary image.

```python 
# Gaussian Mixture Model
from skimage import data
from sklearn.mixture import GaussianMixture
im = data.immunohistochemistry()
img = color.rgb2gray(im)
hist, bin_edges = np.histogram(img, bins=60)
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

#training the Gaussian mixture model
classif = GaussianMixture(n_components=2)
classif.fit(img.reshape((img.size, 1)))

threshold = np.mean(classif.means_)
binary_img = img > threshold
```

Let’s check the results, notice how the cells are nicely separated from the background.

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/segmentation11.webp)

The Bayesian classification approach (or **[Bayesian Maximum Likelihood Classification](https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture5-nb.pdf)**) does not assume that the image contains classes with the same pixel amount (which is true for most cases; where for instance the background contains much more pixels). As an example, let’s suppose we know we have an image with 50 % background, 30% class 1, and 20 % class 2. These are called [_prior probabilities_](https://en.wikipedia.org/wiki/Prior_probability) (since we have prior knowledge before performing our analysis). The Bayesian classification considers this prior knowledge to assign a pixel to a certain class. The Bayes formula says applied to images:

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/segmentation12.webp)

P(ci) is the prior probability (we already know this). P(ci|v) is called the _[posterior probability](https://en.wikipedia.org/wiki/Posterior_probability)_ (in this context, it is the probability of a pixel with a value v belonging to class ci). As seen before, we calculate the probabilities, and the pixel is assigned to the class with a higher probability. P(v) is a normalizing constant and it is set as 1. P(v|ci) is the _[class conditional probability](https://en.wikipedia.org/wiki/Conditional_probability)_ (the probability for a pixel in class ci to have value v) which is calculated using the parametric estimation for each class:

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/segmentation13.webp)

So if we have a pixel of value 100, we calculate the posterior probability for each class for that value (calculating the prior probability and the conditional class probability for each class) and we assign it to a class where the probability is higher. For a pixel and a class, more formally:

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/segmentation14.webp)

With mathematical transformation, the equation can be reduced to a faster version:

![work with color images in python](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/segmentation14.webp)

With the GMM model, we have used an unsupervised approach, where we asked the model to cluster the image in two groups. Despite its existence, Bayesian clustering is much more uncommon than Bayesian classification. We will use **[Naïve Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)** from [Scikit-learn](https://scikit-learn.org/stable/modules/naive_bayes.html) a straightforward and fast implementation. Often, you have images that have been annotated (where the pixels are attributed to different classes and this is your training set) and in this case, you can use the classification model. Naïve Bayes expects we provide the inputs and the labels, in this case, we will use the binary image obtained from GMM as ground truth. The process is very easy in a few lines of Python code we can train our model:

# Additional resources
* [Scikit-image](https://scikit-image.org/)
* [A Study of Image Pre-processing for Faster Object Recognition](https://arxiv.org/abs/2011.06928)




