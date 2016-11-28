# e-cooking

Final Project for IA896 — Introduction to Multimedia Information Retrieval 2016-2 class at FEEC/UNICAMP.

Recognition of the food category and ingredient list of food images. 

####Dataset


<!--figura dos datasets, talvez mais informação sobre eles, proporcao de receitas por ingrediente, etc-->

####Docker

You can find [here](./docker/README.md) the docker images (CPU and GPU) used for the project.

<!--Resultados finais para categoria e ingredientes. Dificuldades, proximos passos, o que deu errado tambem.-->

####Ingredient Recognition
The Convolutional Network VGG16 is used for our ingredient recognition model. Starting from the pre-trained model with ImageNet,
we perform the fine-tuning of the top layers of this pre-trained network with our dataset of recipes, based on the tutorial:

+    [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) (Sections 2 and 3)

The pre-trained weights were obtained from [this](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3) repository.
The pre-processing steps for the VGG16, e.g., mean subtraction, load of images, were obtained from [keras](https://github.com/fchollet/keras).


####References

+ [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)  
+ [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)  