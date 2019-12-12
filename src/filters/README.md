## Training a PP

In order to determine which model should be used for training the PP and for how long, a set of experiments were performed. Results and usage captured below.

### Table of Contents
* Models Tested
* Usage

### Models Tested
* DNN
  * Tested for epochs: {1,2,3,4,5}
* RF
  * Tested for for forests with tree size: {50, 100, 150, 200, 250, 300}

### Usage
In order to run the experiments, perform the following command:
```commandline
   cd <YOUR_EVA_DIRECTORY>
   python pipeline.py
```

The pp has been modified to call sample_train_val instead of split_train_val. The _process_hyper_parameters function iterates through the models and various hyperparameters to perform experiments on how training the same data with different hyperparameters and different models fares. The results are appended to the Output.txt file.
