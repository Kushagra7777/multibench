# Getting Started

## Installation

To install, clone this repository directly from GitHub.

``git clone https://github.com/human-ai-lab/MultiBench.git``

## Python Package Requirements

python >= 3.10

matplotlib >= 3.3.2

memory-profiler >= 0.58.0

numpy >= 1.17.5

pandas >= 1.1.0

scikit-learn >= 0.23.1

scipy >= 1.4.1

torch >= 1.4.0

torchvision >= 0.5.0

tqdm >= 4.36.1

h5py >= 3.3.0 (for robotics data)

PyYAML >= 5.4.1 (for robotics data)

pmdarima >= 1.8.2 (for finance data)

statsmodels >= 0.12.2 (for finance data)

pandas-datareader >= 0.9.0 (for finance data)

## Parts

This repository provides automated end-to-end pipeline that simplifies and standardizes data loading, model construction, experimental setup and evaluation. The repository includes the following parts:

1) **Dataloaders**: We provide scripts for preprocessing and creating dataloaders for the supported datasets under the ``datasets/`` folder. See ``README.md`` for instructions to download the raw data and how to use the data loader scripts for each dataset. 
2) **Training Structures**: Training structures are the "main bodies" that puts the parts of a complete multimodal architecture together and trains them. It takes in all parts of the multimodal architecture (such as unimodal encoders, objective function, fusion module, classification head, etc) as well as hyperparameters as argument. Each training structure scripts provides a train function and a test function. Some training structures are provided under ``training_structures/`` folder.
3) **Unimodal Models**: Unimodal models are convenient modules written for unimodal encoding/decoding. Some unimodal models are provided under ``unimodals`` folder.
4) **Fusion Modules**: Fusion modules are designed to take in encoded representations from multiple modalities and fuses them into a joint representation. Some fusion modules are provided under ``fusions`` folder.
5) **Objective Functions**: The default objective function for classification tasks is ``torch.nn.CrossEntropyLoss`` or ``torch.nn.BCEWithLogitsLoss`` and the default objective function for prediction tasks is ``torch.nn.MSELoss``. However, our pipeline allows customized objective functions for more complex architectures. Some custom objective functions are provided under ``objective_functions`` folder.
6) **Evaluation Scripts**: We also provide scripts for evaluating the performance, complexity and robustness of an architecture on a dataset. These scripts are usually integrated into the training structures (so the performance, complexity and robustness metrics will be automatically computed when a training structure is run).

This modularized design makes the pipeline very easy to use. Below are a few tutorials on usage.

# Tutorial (AV-MNIST Simple Late Fusion)

Here is a simple example of using this repository to quickly build and evaluate a multimodal architecture. We will be using the AV-MNIST dataset. 

The first step is to download the data. You can download the tar file [here](https://drive.google.com/file/d/1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp/view?usp=sharing). Then put it somewhere on your hard drive. Let's assume you put the downloaded tar file under ``~/``. Then you need to untar the file using

```
tar -xvzf avmnist.tar.gz
```

Then you should have a ``avmnist`` folder under ``~/``. Now please change your working directory to the home of this repository. Note that all operations from this pipeline should be done with this working directory. Then you want to write a script to train and evaluate the simple late fusion model on the dataset. At the top of the script, always include the following so that python can find all scripts in this repo:

```
import sys
import os
sys.path.append(os.getcwd())
```

Next, we are going to build dataloaders for AV-MNIST:

```
from datasets.avnist.get_data import get_dataloader
traindata,validdata,testdata = get_dataloader('data/avmnist')
```

Now we have a dataloader each for train, valid and test splits. AV-MNIST is a dataset with energy-reduced MNIST digits and audio of humans reading the digits and the task is to predict the digit using the blurry images as well as the audio.

Now we need to build the multimodal architecture. We first need the unimodal encoders for image and audio modality, respectively. We will use LeNet-3 and LeNet-5 for the image and audio, respectively. First, pick the best available device and construct the unimodal encoders:

```
from utils.device import get_device
from unimodals.common_models import LeNet
device = get_device()  # selects CUDA, MPS, or CPU automatically
encoders=[LeNet(1,6,3).to(device),LeNet(1,6,5).to(device)]
```
Note that the arguments to LeNet are number of input channels, number of internal channels, and number of layers, respectively. Since the input of image modality are 28x28 greyscale pixels and input of audio are 112x112 spectograms, we need 3-layer and 5-layer LeNet respectively. The output of the image encoder will have size 48 and the output of the audio encoder will have size 192. Therefore, if we fuse them by simple concatenation, the joint representation will have size 240. Now let's construct the fusion module and classification head:
```
from fusions.common_fusions import Concat
from unimodals.common_models import MLP
fusion=Concat().to(device)
head=MLP(240,100,10).to(device)
```
Now we have all parts we need! Before training, you can optionally validate that all dimensions line up using `validate_shapes`:
```
from utils.verify import validate_shapes
import torch
validate_shapes(encoders, fusion, head, [torch.zeros(2,1,28,28), torch.zeros(2,1,112,112)])
```
We can start training with Supervised Learning training structure by passing in each part we constructed as well as training and validation data, together some with training hyperparameters (train 30 epochs, use SGD optimizer, with learning rate 0.1 and weight decay of 0.0001):
```
import torch
from training_structures.Supervised_Learning import train
train(encoders,fusion,head,traindata,validdata,30,optimtype=torch.optim.SGD,lr=0.1,weight_decay=0.0001)
```
The training structure will automatically save the model at the epoch with the best validation performance in a file called ``best.pt``. So to test the performance of our saved model, we need to load it and use the test function provided in the same training structure:
```
model=torch.load('best.pt', weights_only=False).to(device)
from training_structures.Supervised_Learning import test
test(model,testdata)
```
That's it! That's all you need to construct, train and evaluate a multimodal architecture with MultiBench.

# Tutorial (AV-MNIST MFM)

MFM is a more complex multimodal architecture compared to the one above. See [this paper](https://openreview.net/forum?id=rygqqsA9KX) for architecture details. But in short, in addition to the supervised cross entropy loss, decoders for each modal are also involved and reconstruction loss is also used in training. Therefore, we will need to use a customized objective function to implement MFM.

We set up the script and get the dataset in the same way as the tutorial above for simple late fusion:
```
import sys
import os
sys.path.append(os.getcwd())

# get data loaders
from datasets.avnist.get_data import get_dataloader
traindata,validdata,testdata = get_dataloader('data/avmnist')
```

But this time we will need both the encoders and decoders for the 2 modalities (image and audio). Suppose we want the latent representataion of each modality to have size 200, then we can write the following:
```
from utils.device import get_device
from unimodals.MVAE import LeNetEncoder,DeLeNet
device = get_device()
n_latent=200
encoders=[LeNetEncoder(1,6,3,n_latent).to(device),LeNetEncoder(1,6,5,n_latent).to(device)]
decoders=[DeLeNet(1,6,3,n_latent).to(device),DeLeNet(1,6,5,n_latent).to(device)]
```
Then we write the fusion layer, which will be a concatenation followed by a MLP:
```
from unimodals.common_models import MLP
from fusions.common_fusions import Concat
from utils.helper_modules import Sequential2
fuse = Sequential2(Concat(),MLP(2*n_latent,n_latent,n_latent//2)).to(device)
```
The MFM architectures requires additional intermediate layers, so we need to construct these as well:
```
intermediates = [MLP(n_latent,n_latent//2,n_latent//2).to(device),MLP(n_latent,n_latent//2,n_latent//2).to(device)]
```
And we also need the classification head:
```
head = MLP(n_latent//2,40,10).to(device)
```
Since MFM's objective function has more components than just one cross entropy loss, we need to use a custom objective function. Luckily, the MFM objective is already provided in MultiBench, so we can directly import and use it:
```
from objective_functions.recon import sigmloss1dcentercrop
from objective_functions.objectives_for_supervised_learning import MFM_objective
objective=MFM_objective(2.0,[sigmloss1dcentercrop(28,34),sigmloss1dcentercrop(112,130)],[1.0,1.0])
```
The arguments passed into the MFM_objective function are weight of cross entropy loss, functions for computing reconstruction loss, and weights for reconstruction loss of each modality. Note that since DeLeNet-3 outputs a 34x34 image but the original input images are 28x28, we will use a ``sigmloss1dcentercrop(28,34)`` to center-crop the 34x34 reconstruction to 28x28 and then apply sigmoid-MSE loss. Similarly, we apply ``sigmloss1dcentercrop(112,130)`` since DeLeNet-5 outputs a 130x130 audio spectogram but the original input spectogram is 112x112. 

Now we have completed all parts of the architecture, we can feed them into the Supervised_Learning training structure to train and test:

```
from training_structures.Supervised_Learning import train,test
train(encoders,fuse,head,traindata,validdata,25,additional_optimizing_modules=decoders+intermediates,objective=objective,objective_args_dict={'decoders':decoders,'intermediates':intermediates})

#testing
model=torch.load('best.pt', weights_only=False).to(device)
test(model,testdata)
```

Note that we need to put the decoders and intermediates as additional optimizing modules, since we want to optimize these modules as well. Also, since we used custom objective function, we need to pass in some arguments to the objective function. The Supervised_Learning training structure will automatically pass in the current encoder output, fusion output, input and training status, but in the case of MFM we also need to give the objective function access to the decoders and intermediates, so we need to add these two to the objective_args_dict so the objective function has access to them.

This is all you need to do for a complex multimodal architecture like MFM!

# Tutorials (creating your own objective function)

Let's say you invented some amazing regularization function ``amazing_reg_fn`` that takes in the list of all modules that matters and returns a regularization loss. Here's how we can write an objective function that returns a weighted sum of Cross Entropy Loss and your regularization loss:

```
def new_objective(regularization_weight):
    def actualfunc(pred,truth,args):
        ce_loss = torch.nn.CrossEntropyLoss(pred,truth)
        reg_loss = amazing_reg_fn(args['all_modules'])
        return ce_loss + regularization_weight*reg_loss
    return actualfunc
```

Note that the ``new_objective`` function creates an actual objective function that takes in 3 arguments: the prediction, the ground truth, and the args dictionary. The args dictionary can contain anything necessary by the objective function. In this case, we need 'all_modules' term in the args dictionary, so we must add that to our objective_args_dict argument in the training structure. For example, if you are using Supervised_Learning training structure, then your train call could look like this:
```
train(encoders,fusion,head,traindata,validdata,25,objective=new_objective(1.0),objective_args_dict={'all_modules':encoders+[fusion,head]})
```
That's all you need to do to create a custom objective function!
