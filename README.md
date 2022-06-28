# automatic-image-captioning

Deep encoder-decoder neural network for the automatic captioning of images. The network is trained on over 400,000 colour images from the [COCO dataset](https://cocodataset.org/), which features common objects in their surroundings and human-generated captions for each image. Below are captions outputted by the network on unseen images after training for one epoch.

<img src="https://github.com/callumcanavan/automatic-image-captioning/blob/master/images/train.png" alt="drawing" width="450"/>

<img src="https://github.com/callumcanavan/automatic-image-captioning/blob/master/images/pizza.png" alt="drawing" width="450"/>

The encoder of this architecture is a pretrained 50 layer CNN (ResNet 50) that takes a 224x224 input image and outputs a flattened feature vector of length embed_size=256 (512 found success in both [[1]](https://arxiv.org/pdf/1411.4555.pdf) and [[2]](https://arxiv.org/pdf/1502.03044.pdf), but I found that this led to my model repeating only one caption after over 8 hours of training). After a batch of these images (batch_size=64 [[2]](https://arxiv.org/pdf/1502.03044.pdf)) are passed through the encoder, their feature vectors are concatenated to one of their corresponding human-generated caption sequences (the words of which are tokenized and embedded in the same size feature space) and used as the input of an LSTM RNN. At each step in the sequence of the RNN, the last embedded vector (starting with the CNN-encoded vector) is used along with the previous state of the LSTM unit to update the hidden and cell states of the unit itself. The unit learns which parts of the input to forget and which to keep, possessing both long and short-term memory of what came before in the input sequence. The hidden and cell state dimensions are 512 in accordance with [[1]](https://arxiv.org/pdf/1411.4555.pdf). Finally, at each step in the sequence, the hidden state goes through a linear layer outputting a vector with length equal to the vocabulary size, the values of which can be used to determine which words are most likely as predicted by the model. The vocabulary was predetermined using the captions in the training data: only words which appeared at least 5 times in the whole set were kept [[1]](https://arxiv.org/pdf/1411.4555.pdf). The vocabulary is loaded from file after its first generation.

All features of the encoder are frozen (except for the last dense embedding layer) to take full advantage of the weights learned by ResNet50 for image classification. 

I completed this project as part of the Udacity Computer Vision nanodegree which provided the blank notebooks and several other functionalities. Algorithm implementations and experiments with parameters/inputs were completed by me.

[1] [Show and Tell: A Neural Image Caption Generator, Vinyals et al., 2014](https://arxiv.org/pdf/1411.4555.pdf)

[2] [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention, Xu et al., 2015](https://arxiv.org/pdf/1502.03044.pdf)

# Depends
```
pytorch cv2 numpy
```
