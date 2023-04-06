# Models for Infinite Dimensions (MID)

<img src=https://github.com/Bella-cell/VAE/blob/main/doc/MIDLogo.png width=300 p align="right">

The MID project aims to produce a machine learning toolkit that can return the physical meaning behind the complex dataset. Considering the continuously growing research interest in using high throughput experiments in characterizations of newly synthesized samples/materials, tons of spectra data are being conducted, and thus an efficient toolkit for analyzing data is in great needed. In this project, the toolkit is built up with several unsupervised analytical methods, including normal PCA, tangent PCA and VAE. These methods will lower the dimension of a given dataset and produce plots of new low-dimensional spectrum. Therefore, it will make it easier for users to understand the physical meaning of the originally complex data. Moreover, by adding and subtracting one standard deviation of each newly generated spectrum, the user will be able to observe more information of the data and even make prediction for their future experiments.

-----
## Software Dependencies
For those who would like to run the jupyter and python files, please ensure you have the following:
- Python 3.7
- Python packages listed in `environment.yml`and Installation section.

-----
## Installation
Install and activate the 'MID' environment in your desired directory with the following commands:

`git clone ......`

`cd MID`

`conda env create -f environment.yml`

`conda activate MID`

This enviroment contains the following packages: <br>
- jupyter
- pandas
- numpy
- scikit-learn
- pip
- pip:
  - plotly==5.6.0

-----
## Organization


- UW Chemical Engineering<img src=https://github.com/Bella-cell/VAE/blob/main/doc/organization_iamge.png width=600 p align="right">
- Clean Energy Institute 
- UW Direct           



-----
## spectrum Data
The dataset is obtained from UV-vis spectroscopy of a kind of material in prof. Lilo Pozzo’s laboratory. Since the dataset is composed of 448 samples with 101 features, the original dimension of the dataset in 101, which is too complex to analyze and need further dimension reduction. Moreover, the data are functional data, which means they couldn’t be simply express by vectors and need more advanced analytical methods such as tangent PCA and VAE to process them.
To see how spectrum data was obtained, please see `data` directory.

Citation for accompanying publication:
[Link](https://pubs.rsc.org/en/content/articlelanding/2022/dd/d2dd00017b)

-----

## Authors
Bella Wu, Material Science and Engineering <br>
Kim, Material Science and Engineering <br>
Lilo Yeh, Material Science and Engineering <br>
Nick Leu, Material Science and Engineering <br>
William Lin, Material Science and Engineering <br>
