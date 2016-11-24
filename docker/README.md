## **Docker**

We have created two docker images for our experiments (cpu-based and gpu-based).
First, you must install Docker from [here](https://www.docker.com/what-docker). 

**Images**

+ **CPU**: `kbogdan/docker_dl_up:cpu`
    This image was created from the image provided by [saiprashanths](https://github.com/saiprashanths/dl-docker). In our version TensorFlow, Keras, and other libraries were updated (e.g. Numpy). Update: nltk installed. 
+ **GPU**:  `kbogdan/cuda-keras-tf:latest`
    This image was created from [here](https://github.com/Kaixhin/dockerfiles/tree/master/cuda-keras/cuda_v7.5). In our version, TensorFlow and other libraries were installed (e.g. Numpy, SciPy, Cython, nltk). Update: matplotlib installed.

Once Docker is installed, you can follow the steps:

1. Get the image, where `image_name:tag` is one of the above options:
    > docker pull image_name:tag
  
2. Create a new container.

    **CPU**
    > docker run -it -p 8888:8888 -p 6006:6006 -v /your-local-folder:/root/sharedfolder image_name:tag bash
    
    **GPU**
    > nvidia-docker run -it -p 8888:8888 -p 6006:6006 -v /your-local-folder:/root/sharedfolder image_name:tag bash
    
   /your-local-folder:/root/sharedfolder means that you are going to share a folder (your-local-folder) with the container. This folder in the container will be: /root/sharedfolder.
   

In order to change the backend of Keras to Theano you can change the property 'backend' in the file `~/.keras/keras.json` from 'tensorflow' to 'theano'.
Configuration of `~/.keras/keras.json` in both docker images:
>{
>"image_dim_ordering": "th",
>"epsilon": 1e-07,
>"floatx": "float32",
>"backend": "tensorflow"
>}


TODO version of all libraries in both images