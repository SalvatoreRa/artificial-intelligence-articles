# Introduction to medical image analysis

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


