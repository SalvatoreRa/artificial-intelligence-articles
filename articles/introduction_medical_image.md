# Introduction to medical image analysis

## Index

* [Introduction to medical imaging](#Introduction-to-medical-imaging)
  * [Modalities](#Modalities)
  * [Framework](#Framework)
  * [Deep learning in medical image analysis](#Deep-learning-in-medical-image-analysis)
  * [Summary of the section](#Summary-of-the-section)
* [Introduction to point processing](#Introduction-to-point-processing)

## Introduction to medical imaging

In general, we can say that the primary scope of radiological imaging is to produce images, which depict anatomy or physiological function well below the skin surface.

Different types of medical images are produced by varying the types of energies used to acquire the image. These different modes are called radiology modalities. Different modalities present different aspects (time of exposition, scanning methods, use of radioactive isotopes). As an example, we prefer images that can be acquired in a short time. However, sometimes this is not possible, like in nuclear medicine where you use radioactive isotypes (you need to inject them in the patient and wait for them to diffuse, and since the time of decay is decided by the physics it can require minutes to achieve an image). A slower method has its drawback: the patient has the involuntary motion of the lung, heart, and esophagus over this time frame, thus long time to scan leads to a lower resolution.

The goals of medical image analysis are:

* Quantification, measuring the features of a medical image (like area or volume)
* Segmentation, which is a step used in general to make features measurable (you segment an object and you measure the properties)
* Computer-aided diagnosis, given measurements and features makes a diagnosis.

### Modalities

* **Radiography**. It was the first medical imaging technology, thanks to the physicist Wilhelm Roentgen who discovered X-rays on November 8, 1895 (he made the first radiography of the human body when produced an image of his wife’s hand). Radiography is performed with an X-ray source on one side of the patient and an X-ray detector on the other side (a short-duration pulse of X-rays is emitted by the X-ray tube, it passes by the patient and arrives on the detector producing the image). The interaction with the patient is scattered and that is recorded in the image (this phenomenon is called attenuation). The attenuation properties of the anatomic structures inside the patient such as bone, liver, and lung are very different, allowing us to have an image of these tissues.
* **Mammography** is a radiography of the breast. Since the low subject contrast in breast tissues, the technique uses much lower x-ray energies than radiography, and thus the x-ray and detector systems are different and specifically designed for breast imaging. It is used to routinely evaluate asymptomatic women for breast cancer.
* **Computer tomography** was a breakthrough in medicine during the 70s eliminating the use of exploratory surgery. CT images are obtained by acquiring numerous (around 1000) X-ray projection images over a large angular swath by the rotation of the X-ray source and detector. The acquired images are then reconstructed with an algorithm. CT results in high-resolution thin-slice images of an individual (moreover allowing a 3D reconstruction). Moreover, there is no superimposition of the anatomical structure allowing a better interpretation. The CT image data set can be used to diagnose the presence of cancer, ruptured disks, subdural hematomas, aneurysms, and other pathologies. There is a balance between radiation dose and image resolution, mathematical techniques and artificial intelligence are helping to increase the resolution keeping the same dose. In addition, the introduction of contrast can be used to study vascularity and perfusion of organs
* **Magnetic resonance imaging**. MRI uses magnetic fields that are about 10,000 to 60,000 times stronger than the Earth’s magnetic field. MRI utilizes the nuclear magnetic resonance (NMR) properties of the nucleus of the hydrogen atom.
* **Fluoroscopy**. It uses X-ray detector systems capable of producing images in rapid temporal sequence. Fluoroscopy is used for positioning intravascular catheters, visualizing contrast agents in the GI tract, and image-guided intervention including arterial stenting.
* **Ultrasound**. It uses high-frequency sound waves, that are reflected off tissue to develop images of joints, muscles, organs, and soft tissues. It is considered the safest form of medical imaging and is used in a wide range of cases
* **Positron emission tomography**. PET is a functional imaging technique that uses radioactive substances known as radiotracers to visualize and measure changes in metabolic processes. It is extensively used in oncology to research metastasis. However, it is also used in brain imaging for research on seizures.

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

* Spatial resolution: number of pixels for representing an image
* Gray level quantization is the number of gray levels to represent the image. Typical gray resolutions are 8, 10, and 12-bit (8-bit is corresponding to 256 levels). In medical images, there is an associated physical measurement to these gray levels (for instance in computer tomography, a pixel value represents the Hounsfield Units (HU))

**Digital images**

The image’s content is transformed into a pixel (with a value range from 0 to 255, defined as the intensity of the pixel). A gray-scale image is just a 2D matrix with m x n pixels (in this case 0 represents black and 255 pure white). A color image is defined by three matrices (one for each channel in the RGB format) and therefore the format of m x n x c. The total number of pixels (m x n) defines the size of the image (for instance for a sensor this is the maximum image it can acquire). We can have also multi-spectral images or multi-channel images (ex. Satellite images that are integrated with other wavelengths like infrared).

A special case is binary images, where 0 represents the background and 1 the foreground (normally, these images are the results of thresholding and other algorithms). Another special case is label images, where pixels are associated with a number, representing to which object they belong (for example a label image can have a value of 0 for the sky, 1 for each pixel part of a human figure, 2 for vehicles, and so on). Normally, Label images are obtained after image segmentation (or hand-annotated as an example to train these algorithms).

Generally, in programming languages, the image is converted to an array (a matrix m x n) and the first pixel has generally coordinates 0,0 (starting from the upper left corner). These coordinates are used also in the plotting. Programs are conducting mathematical operations on each point of the matrix (pixel), with large images this can be computationally expensive.

Moreover, not all the image is interesting for further analysis, it is often common to define a region of interest (ROI) which is generally a common rectangle containing the pixel of interest. As a simple example, if you want to count the daily car numbers in the mall’s parking over one year, your ROI in the image would be the zone encompassing the parking area.

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
