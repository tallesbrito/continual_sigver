# Robust Handwritten Signature Representation with Continual Learning of Synthetic Data over Predefined Real Feature Space

This repository contains model and inverted data that can be applied to perform knowledge distillation about real offline signature characteristics to other backbones or architectures.

The method by which the data was inverted from the SigNet model is described in the paper "Robust Handwritten Signature Representation with Continual Learning of Synthetic Data over Predefined Real Feature Space" submitted to the International Conference on Document Analysis and Recognition.

# Usage

## Reading inverted data

Inverted data is provided in a single .npz file, with the following components:

* ```x```: Inverted examples (numpy array of size N x 1 x H x W).
* ```y```: The respective user of the inverted example (numpy array of size N).

The inverted examples have a 150x220 size, which is the input size of the SigNet model.

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

