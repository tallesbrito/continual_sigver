# Expanding Generalization of Handwritten Signature Feature Representation through Data Knowledge Distillation

This repository contains model and inverted data that can be applied to perform knowledge distillation about real offline handwritten signature characteristics to other backbones and architectures.

The method to perform distillation from SigNet model is described in the paper:  T. B. Viana, V. L. F. Souza, A. L. I. Oliveira, R. M. O. Cruz, and R. Sabourin, “Robust handwritten signature representation with continual learning of synthetic data over predefined real feature space,” in Document Analysis and Recognition - ICDAR 2024, E. H. Barney Smith, M. Liwicki, and L. Peng, Eds. Cham: Springer Nature Switzerland, 2024, pp. 233–249.

An extended analysis of the proposed method is currently submitted to the IEEE Transactions on Information Forensics & Security journal.

# Usage

## Data preprocessing

The functions in this package expect continual training data to be provided in a single .npz file, with the following components:

* ```x```: Signature images (numpy array of size N x 1 x H x W)
* ```y```: The user that produced the signature (numpy array of size N )
* ```yforg```: Whether the signature is a forgery (1) or genuine (0) (numpy array of size N )

It is provided functions to process some commonly used datasets in the script ```sigver.datasets.process_dataset```. 

As an example, the following code pre-process the MCYT dataset with the procedure from [1] (remove background, center in canvas and resize to 170x242)

```bash
python -m sigver.preprocessing.process_dataset --dataset mcyt \
 --path MCYT-ORIGINAL/MCYToffline75original --save-path mcyt_170_242.npz
```

During training a random crop of size 150x220 is taken for each iteration. During test we use the center 150x220 crop.

## Reading inverted data

Inverted data is provided in a single .npz file, with the following components:

* ```x```: Inverted examples (numpy array of size N x 1 x H x W).
* ```y```: The respective user of the inverted example (numpy array of size N).
* ```yforg```: All examples are genuine (0) (numpy array of size N).

The inverted examples have a 150x220 size, which is the input size of the SigNet model. You should not apply any transformations (crop, normalization, etc.) to this data.

## Distilling knowledge with the SigNet model and inverted data

In order to distill knowledge from SigNet while incrementally learning another dataset through continual learning, you should inform the incremental dataset <continual-data.npz> with a defined user range [first last]. The intensity of SigNet distillation in the obtained continual representation space is defined by the --p-lamb argument as follows:

```
python -um sigver.featurelearning.distill --s-model signet \
  --c-dataset-path <continual-data.npz> --c-users [first last] \
  --p-lamb 1.7 --logdir continual_model
```

## Training WD classifiers

For training and testing the WD classifiers, use the ```sigver.wd.test``` script. Example:

```bash
python -m sigver.wd.test --model-path <path/to/trained_model> \
    --data-path <path/to/data> --save-path <path/to/save> \
    --exp-users 0 300 --dev-users 5000 7000 --gen-for-train 12
```
The parameter ```--gen-for-train``` defines the number of reference signatures for each user of the exploitation set in WD approach.

The example above train WD classifiers for the exploitation set (users 0-300) using a development
set (users 300-881), with 12 genuine signatures per user (this is the setup from [1] - refer to 
the paper for more details). 

For knowing all command-line options, run ```python -m sigver.wd.test```.

## Training WI classifiers

For training and testing the WI classifiers, use the ```sigver.wi.test``` script. Example:

```bash
python -m sigver.wi.test --model-path <path/to/trained_model> \
    --data-path <path/to/data> --save-path <path/to/save> \
    --exp-users 0 300 --dev-users 5000 7000 --gen-for-ref 12
```

The parameter ```--gen-for-ref``` defines the number of reference signatures for each user of the exploitation set in WI approach.

The example above train a WI classifier for the exploitation set (users 0-300) using a development
set (users 300-881). The WI classifier is tested with 12 reference signatures per user (this is the setup from [2] - refer to the paper for more details). 

For knowing all command-line options, run ```python -m sigver.wi.test```.

# Evaluating the results

When training WD or WI classifiers, the trained_model is a .pth file (a model trained with the script above, or pre-trained - see the section below). These scripts will split the dataset into train/test, train WD/WI classifiers and evaluate then on the test set. This is performed for K random splits (default 10). The script saves a pickle file containing a list, where each element is the result  of one random split. Each item contains a dictionary with:

* 'all_metrics': a dictionary containing:
  * 'FRR': false rejection rate
  * 'FAR_random': false acceptance rate for random forgeries
  * 'FAR_skilled': false acceptance rate for skilled forgeries
  * 'mean_AUC': mean Area Under the Curve (average of AUC for each user)
  * 'EER': Equal Error Rate using a global threshold
  * 'EER_userthresholds': Equal Error Rate using user-specific thresholds
  * 'auc_list': the list of AUCs (one per user)
  * 'global_threshold': the optimum global threshold (used in EER)
* 'predictions': a dictionary containing the predictions for all images on the test set:
  * 'genuinePreds': Predictions to genuine signatures
  * 'randomPreds': Predictions to random forgeries
  * 'skilledPreds': Predictions to skilled forgeries


## Download model and data for distillation

* Pre-trained SigNet model for distillation: ([link](https://drive.google.com/file/d/14FNyw5ay1PLqB_jRkKvhkylj51LDZ08B/view?usp=drive_link))
* Inverted data using deep inversion with competition: ([link](https://drive.google.com/file/d/1El-9S-RKGMBmYy4G26dtvKEgAYoX2xpI/view?usp=drive_link))

