## Documentation

### This document provides instructions on how to train an ECG encoder using a bi-branch self-supervised ECG architecture and how to use the trained ECG encoder to transform ECG signals for other ECG-related applications.
### This study uses three ECG datasets for model training: PTBXL, Chapman, and CPSC, resulting in three corresponding trained models. All datasets are open source. The sources of the datasets used in this study are as follows:
* PTBXL：https://physionet.org/content/ptb-xl/1.0.3/
* Chapman：https://physionet.org/content/ecg-arrhythmia/1.0.0/
* CPSC：http://2018.icbeb.org/Challenge.html

The steps to use the ECG encoder are as follows:
* Training the ECG encoder
    1. Download the desired ECG dataset.
    2. Create a virtual environment using conda according to the requirement.txt file.
    3. Configure the dataset path, GPU ID, save location, and other desired model settings in run_experiment.py.
    4. Run python run_experiment.py to start the training process.
* Using the ECG encoder
    1. Set the correct model parameters and model save location.
    2. Package the ECG data to be transformed into the torch.utils.data.DataLoader class, ensuring the output dimensions are (BatchSize, Channels, Length).
    3. Use model.encode(DataLoader) to perform feature transformation.

## Acknowledgement

This project was supported by Ministry of Science and Technology, Taiwan, under grant no. <font color=#FF0000>MOST ???.</font> 
