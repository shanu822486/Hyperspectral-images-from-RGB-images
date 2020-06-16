# Hyperspectral-images-from-RGB-images
Our main aim in this project is the recovery of whole-scene hyperspectral (HS) information from a 3-channel RGB image.

In which we are working on two types track — —

1.Real World images(Images form by an unknown camera and saved in a lossy image format)
2.Clean images(Images form by the spectrally-calibrated system)

Our Proposed Method:

The NTIRE 2018 spectral reconstruction challenge proposed a high level and uniform benchmark for HS-from-RGB systems.

Take the reference of Research paper-

http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Arad_NTIRE_2018_Challenge_CVPR_2018_paper.pdf

We try different models to achieve more accurate results.

1-HS-ResidualNet Model

2-Attention HS-ResNet Model

3-Advanced CNN-Dense Net Model

4- UCNN-D Model




Dataset:

   The dataset is divided into:

   Train data 1: 203 hyperspectral images from the BGU ICVL Hyperspectral Database.
   Train data 2: 53 newly collected hyperspectral images.
   Validation data: 10 Hyperspectral images collected alongside “Train data 2”, their corresponding RGB images are also provided.
   Test data: 20 Hyperspectral images (10 for each track) collected alongside “Train data 2”.





References:

1. http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Arad_NTIRE_2018_Challenge_CVPR_2018_paper.pdf
2.http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Shi_HSCNN_Advanced_CNN-Based_CVPR_2018_paper.pdf

3. https://competitions.codalab.org/competitions/18034#participate-getdata
