# SSAFNet-README

The code is the official implement of SSAFNet, which was designed for EEG depression recognition. We proposed SSANet to extract the time-space domain feature of the EEG data and employed diffential entropy to extract the frequency domain feature. Finally we designed a fusion model to combine two domains' feature. Read the code for details.

## Paper Status
This repository contains code developed to support a scholarly paper currently under review. Details and specific results of the paper will be disclosed once the review process is complete.

## Installation
If `git` is avaliable, installation can be done via
```
git clone https://github.com/Wonder-How/SSAFNet
```

To control thte version of code, you can use the `requirements.txt` as followed

```
pip install requirements.txt
```

## Data preparation

The SSAFNet will employ two types of data. One is the differential entropy characteristics of EEG and the other one is the EEG timing signals. 

The EEG differential entropy information is stored in a csv file, and the first column identifies whether depression is 1, depression is 0, normal is 0. The next 15 are listed as differential entropy data, and columns 2-6, 7-11, and 12-16 are the information of three leads respectively, and the differential entropy of each lead is divided into five bands, namely delta,theta,alpha,beta, and gamma bands.

The EEG timing data used in the model is divided into six parts, namely the original data and the filtered information of five frequency bands. The csv file is organized the same for each of the data. The first three columns are subject, experiment round, and depression marker (subject and experiment round were not used in this model). The next 3000 columns are timing data for Fp1,Fpz, and Fp2, with 1000 data points per lead.

If your data is organized differently, change the code `datasets.py`.

## Usage

In the main function, follow your data path. To better train the code, you can use pre-trained weights for both the time domain model and the frequency domain model. Provides freeze function to speed up model training. Then

```
python main.py
```

We also provide the code of importance ranking of differential entropy features, which can be referred to as the following code`MLP_Feature_Importance.py`.

## Contact Information

If you have any questions about the code or research methods, please contact us via:

- Email: [wonderhow@bit.edu.cn](mailto:wonderhow@bit.edu.cn)

We encourage other researchers to use and extend these methods, and we look forward to your valuable feedback.
