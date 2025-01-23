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

## Introduction to medical imaging

In general, we can say that the primary scope of radiological imaging is to produce images, which depict anatomy or physiological function well below the skin surface.

Different types of medical images are produced by varying the types of energies used to acquire the image. These different modes are called [radiology modalities](https://ccdcare.com/resource-center/radiology-modalities/). Different modalities present different aspects (time of exposition, scanning methods, use of radioactive isotopes). As an example, we prefer images that can be acquired in a short time. However, sometimes this is not possible, like in nuclear medicine where you use [radioactive isotypes](https://world-nuclear.org/information-library/non-power-nuclear-applications/radioisotopes-research/radioisotopes-in-medicine) (you need to inject them in the patient and wait for them to diffuse, and since the time of decay is decided by the physics it can require minutes to achieve an image). A slower method has its drawback: the patient has the involuntary motion of the lung, heart, and esophagus over this time frame, thus long time to scan leads to a lower resolution.

The goals of medical image analysis are:

* Quantification, measuring the features of a medical image (like area or volume)
* Segmentation, which is a step used in general to make features measurable (you segment an object and you measure the properties)
* Computer-aided diagnosis, given measurements and features makes a diagnosis.

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

An image without compression (256 x 256) pixels with 1 byte (256 levels) has a size of 65 kb. But raw images can be extremely large and a dataset can be composed of thousands of images (with a large cost of storage). A compression algorithm aims to represent an image with fewer pixels. Since many neighbor pixels present the same color (or sale value) an image can be easily compressed. This is the principle of the simplest compression algorithm Run-Length Encoding (RLE), where for the pixels with the same values we store only the value and the count of the pixels with the same value. After running the algorithm we can calculate the compression ratio (uncompressed size/ compressed size). Notice, that RLE does not lead to loss of information (or lossless compression). But not all the compression algorithms are lossless, while PNG is lossless JPEG is lossy.

Choosing the compression level is important because high compression leads to artifacts in the image. Indeed, the lossy image format should be avoided so as to not introduce artifacts in the analysis.

**DICOM format**

A common image format for medical imaging (it contains additional information in the header such as hospital, patient, scanner, image information, and anatomy axis). For CT or PET scans, each slice is an image and the patient can be scanned multiple times (DICOM format contains information about the series). PyDicom is a popular package for analyzing DICOM images in Python. It can be lossy or lossless.

### Deep learning in medical image analysis

Artificial intelligence (AI) and machine learning, especially deep learning, applications are used extensively in everyday life and in many domains (from cars to internet apps). There is extensive literature about deep learning and medical image analysis since this is an active field of research. However, there are not so many algorithms that are used in hospitals and healthcare. In fact, there is a lot at stake in the medical imaging field, since an algorithm can have serious consequences (missing a cancer diagnosis, leads to delayed treatment, and potential loss of life).

Just as a brief introduction, the difference between deep learning and traditional machine learning techniques is that the former can automatically learn meaningful representations of the data (this means you do not need a lot of preprocessing to extract features by yourself). However, deep learning requires a lot of data, is computationally expensive, and requires sophisticated hardware. Moreover, many deep learning models have been trained and tested on small images (64x64, 256x256) while medical images are often in high resolutions (or we can 3d or even 4d). In addition, the medical images training set needs to be annotated by a medical expert who will establish the ground truth (which means a lot of work from an expert if your dataset is quite large). There are also problems with artifacts, different machines used to acquire the images, and so on.

There are many tasks and challenges where deep learning can be useful in medical image analysis. It is interesting to discuss a few of them just as an introduction.

For example, prior to deep learning, most medical image segmentation methods relied on algorithms like thresholding, clustering, and other human-designed algorithms. Deep learning instead needs just an example to generate its own algorithm and to discriminate and proceed to segmentation. Indeed, segmentation is a tedious and long process to conduct by hand, thus image segmentation is an active field of research and playground for deep learning. Image segmentation is important to select potential cancer lesions, and metastasis, study blood vessels, and so on. A different but similar task is object localization (there are two kidneys in the body, segmentation is to select the kidney pixels while localization is to identify the two different organs). Moreover, the extraction of features (radiomics) can be fundamental to creating predictive models.

However, another important challenge in the use of deep learning in clinics is that often these models are a black box. Explainability is fundamental to use in daily healthcare, we need to know why a model is predicting a certain outcome, or what drives its decision when identifying in the image a region as abnormal.

### Summary of the section

The world of biomedical imaging is a fascinating one, but it also presents complex challenges. Given the importance of these types of data for both diagnostics and countless applications, countless groups and models have been developed. In subsequent articles, we will address the basics of being able to analyze images.

## Introduction to point processing

When you have used an Instagram filter to correct brightness, without noticing you are doing point processing. In simple words, you have an input image f(x,y) and you want to produce an output image g(x,y). In point processing, we are conducting an operation on the image in the way a pixel now has a new image. In comparison to other processing functions, here the value of the pixel in the output is based on the value of the pixel in the input image. The name point processing derives from the fact that the neighbor pixels have no role in the transformation.

### Histogram

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/Example_histogram.png)
*from Wikipedia*

A histogram is a graphical representation of the data that organizes a group of data points into user-specified ranges. In a way, looks like a bar graph, but differently from the bar graph is a more sophisticated plot that condenses a data series: grouping data into logical ranges or bins. Histograms are primarily used in statistics, for instance in demography someone can use a histogram to show how many people are between a certain age range (0–10, 11–20, 21–30, and so on). The **range of bins** is decided by the user: we can divide by an arbitrary number of bins (for instance, in the above example we used bins of width 10, but we could have decided the interval was 5 years).

In general, we plot the **frequency** on the y-axis (if we have 655 individuals in the interval 0–10, the corresponding bar would have a value of 655). However, we can plot the **percentage of the total or density**. Despite many people using histograms and bar charts as interchangeable terms, histograms provide the frequency distribution of variables while bar charts are more indicative to represent a graphical comparison of discrete or categorical variables. In fact, histograms are in general used with continuous variables.

Histograms have shown to be valuable in many cases: different types of data, time series where you group data for hours and show the frequency on the y-axis, and also images. Histograms are very useful when you are interested in the general distribution of your features (to check if the distribution is skewed, symmetric or there are outliers).


A few suggestions for using histograms:

* Variables should be continuous since a histogram is meant to describe the frequency distribution of a continuous variable and thus is a misuse if you use a categorical variable.
* Use a **zero-valued baseline**, since the height of each bin represents the frequency of the data changing the baseline (starting from an arbitrary height, insert a gap) is altering the perception of the data distribution
* The choice of the number of bins is important. In general, there are some rules of thumb but domain knowledge is normally driving the choice (and better try some options and see which one is leading to the best result). Remember that the bin size is normally in inverse relationship with the number of bins (the larger the bins, the fewer cover the data distribution, however choosing too many bins is a bad choice as well). In fact, if you have too many bins the data distribution looks rough (and hard to understand) but too few, you do have not enough details. Many histogram implementations use some algorithms to find the appropriate number of bins.
* Bin size can be also of unequal size in case of sparse data (but this is dangerous water and you should avoid it)

**Image histogram**

The image histogram (value of the pixels on the horizontal axis and frequency on the vertical axis) allows for inspecting the image. The histogram represents the tonal distribution of a digital image (the number of pixels in an image in the function of their intensity), allowing us to inspect the distribution. If most of the pixels are close to the left, the image would be dark. Conversely, if the curve is too close to the right edge, the image will be too bright. Histogram peaks can be informative to retrieve a particular structure in the image and can be used to select them through the threshold. Moreover, can be used to adjust contrast, brightness, or tonal value.

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/image_instogram.webp)

The histogram of an image can be described as a function h(v) where h is the frequency of pixels with value v, where the total pixel number is N

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/image_instogram2.webp)

Which can be normalized by dividing by N

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/image_instogram3.webp)

In this case, H(v) represents the probability for a pixel to have value v:

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/image_instogram4.webp)

The cumulative histogram is the progressive sum for each bin of the pixel frequencies (if for a histogram the frequencies for the first 3 bins are 2, 3, 6 for the cumulative is 2, 5, 11). More formally. For a histogram with n bins (an arbitrary number between 0 and 255) with height H(n) the cumulative histogram:

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/image_instogram5.webp)

I will show here how to load an image with Python (there are many ways, but this is easier with Google Colab). The image is also resized to be easier to handle.

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

Brightness modification is changing the value of each pixel by adding a constant c (or subtracting)

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

Contrast correction is meant to increase the separation of pixels with close value by multiplying for a constant c. In this case, the slope of the function is modified.

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

Some algorithms require that the image is in a certain range (ex. 0–1). In this case, we have to remap to a different range the pixels. This is the general equation, where Vmax and Vmin are the actual maximum and minimal values of the image, and Vmax’ and Vmin’ are the desired extremes of the new range:

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

**Histogram stretching** is a technique to adjust contrast and brightness at the same time. The idea is to stretch the image histogram so very dark and very bright bins are used (i.e. near 0 and near 255). Considering a histogram with the minimum value of f1 and the maximum value of f2 (ex. An image with a pixel value range 20–180 has f1= 20 and f2 = 180), histogram stretching can be obtained:

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing12.webp)

For our example:

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing13.webp)

Histogram stretching enhances contrast and enables better recognition of small details, however, is sensible to outliers. Therefore, a more sophisticated technique of **histogram equalization** can be used. This transformation is based on the cumulative histogram.

There are also derived algorithms as **adaptative histogram equalization** and **Contrastive Limited Adaptive Equalization** which avoid noise amplification.

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

**Gamma mapping** is defined by elevating to γ constant the f(x,y). The value of γ < 1 increases the dynamics in dark areas (the mid values are increased, basically the gray pixels are affected), why γ > 1 increase the bright area (the mid pixel values are decreased)

![description of halogen and example of image modalities ](https://raw.githubusercontent.com/SalvatoreRa/artificial-intelligence-articles/refs/heads/main/images/point_processing15.webp)

**Logarithmic mapping** is using the logarithmic function to enhance pixels with low intensity. Generally, it is used when there are few bright spots on a dark background or when the dynamic range is large.

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



# Additional resources
* [Scikit-image](https://scikit-image.org/)


