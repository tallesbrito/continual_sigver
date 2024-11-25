# Expanding Generalization of Handwritten Signature Feature Representation through Data Knowledge Distillation

This repository contains model and inverted data that can be applied to perform knowledge distillation about real offline handwritten signature characteristics to other backbones and architectures.

The method to perform distillation from SigNet model is described in the paper:  T. B. Viana, V. L. F. Souza, A. L. I. Oliveira, R. M. O. Cruz, and R. Sabourin, “Robust handwritten signature representation with contin- ual learning of synthetic data over predefined real feature space,” in Document Analysis and Recognition - ICDAR 2024, E. H. Barney Smith, M. Liwicki, and L. Peng, Eds. Cham: Springer Nature Switzerland, 2024, pp. 233–249.

An extended analysis of the proposed method is currently submitted to the IEEE Transactions on Information Forensics & Security journal.

# Usage

## Reading inverted data

Inverted data is provided in a single .npz file, with the following components:

* ```x```: Inverted examples (numpy array of size N x 1 x H x W).
* ```y```: The respective user of the inverted example (numpy array of size N).

The inverted examples have a 150x220 size, which is the input size of the SigNet model. You should not apply any transformations (crop, normalization, etc.) to this data.

## Distilling knowledge with the SigNet model and inverted data

In order to distill knowledge from SigNet, you pass the inverted data through SigNet and obtain the output probabilities of the classification layer (```signet_logits```). The same inverted data is passed in the student model, and the student output probabilities are obtained (```student_logits```). The divergence between models can be calculated using the Kullback-Leibler divergence as follows (in our experiments, we adopted the temperature hyper-parameter as 1.0):

```
import torch
from torch.nn import functional as F
...
kl_loss = torch.nn.KLDivLoss(reduction='batchmean').to(device)
T = 1.0
divergence_loss = kl_loss(F.log_softmax(student_logits / T, dim=1), F.softmax(signet_logits / T, dim=1))

```

## Download model and data for distillation

* Pre-trained SigNet model for distillation: ([link](https://github.com/tallesbrito/continual_sigver/blob/master/models/signet.pth))
* Inverted data using deep inversion with competition: ([link](https://github.com/tallesbrito/continual_sigver/blob/master/data/data.npz))

