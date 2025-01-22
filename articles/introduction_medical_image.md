# Introduction to medical image analysis

## Introduction to medical imaging

In general, we can say that the primary scope of radiological imaging is to produce images, which depict anatomy or physiological function well below the skin surface.

Different types of medical images are produced by varying the types of energies used to acquire the image. These different modes are called radiology modalities. Different modalities present different aspects (time of exposition, scanning methods, use of radioactive isotopes). As an example, we prefer images that can be acquired in a short time. However, sometimes this is not possible, like in nuclear medicine where you use radioactive isotypes (you need to inject them in the patient and wait for them to diffuse, and since the time of decay is decided by the physics it can require minutes to achieve an image). A slower method has its drawback: the patient has the involuntary motion of the lung, heart, and esophagus over this time frame, thus long time to scan leads to a lower resolution.

The goals of medical image analysis are:

* Quantification, measuring the features of a medical image (like area or volume)
* Segmentation, which is a step used in general to make features measurable (you segment an object and you measure the properties)
* Computer-aided diagnosis, given measurements and features makes a diagnosis.

