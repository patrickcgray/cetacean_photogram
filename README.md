# Drones and Convolutional Neural Networks Facilitate Automated and Accurate Cetacean Species Identification and Photogrammetry

This repository has the code needed to go along with the Methods in Ecology and Evolution manuscript "Drones and Convolutional Neural Networks Facilitate Automated and Accurate Cetacean Species Identification and Photogrammetry."

## Environment Setup
The environment can be set up quite simply for this project using Docker. The Dockerfile included in this directory will create a functional environment matching what was used for training, validation, and testing in the paper. Once Docker is running on your machine this can be created with the se commands

Build the image:
```
docker build -t photogram_image .
```

Create a container using this image, have it access your NVIDIA runtime, start up a jupyter notebook, and expose that port and the port needed for running tensorboard:

```
docker run --name photogram_container --runtime=nvidia -it -p 8888:8888 -p 6006:6006 -v ~/:/host photogram_image jupyter notebook --allow-root --ip 0.0.0.0 /host
```

Once this container is running you can stop it either by exiting the open terminal or:

```
docker stop photogram_container
```

This can then be restarted with:

```
docker start photogram_container
```

And once this is running you can re-access the terminal of this container with:

```
docker attach photogram_container
```

With the Docker container running you can simply go to your browser at the address `http://127.0.0.1:8888`  and view all the jupyter notebooks included in this repository needed for training, testing, and applying this model. 

This code works alongside with the Matterport implementation of [Mask RCNN](https://github.com/matterport/Mask_RCNN/) and requires that to be in a directory of the same level as this repository. For installing that repository we will refer you to the instructions in that [github repo](https://github.com/matterport/Mask_RCNN#installation) and you can use those simple installation instructions from within this Docker container.

This code also expects there to be a directory named `photogram_data` at the same level as this directory and that is where is pulls data. So your structure should look like

```
<overarching_dir>/
  Mask_RCNN/
  photogram_data/
  cetacean_photogram/
```


## Run Jupyter notebooks

Open the `inspect_whale_data.ipynb` or `inspect_whale_model.ipynb` Jupter notebooks. You can use these notebooks to explore the dataset and run through the detection pipelie step by step.

## Train the model

If you are training a new model from scratch it is recommended to start from pre-trained COCO weights. You can download those at: https://github.com/matterport/Mask_RCNN/releases/tag/v2.0 and then use the command:

```
python whale.py train --dataset=/path/to/dataset --weights=coco
```

Resume training a model that you had trained earlier

```
python whale.py train --dataset=/path/to/dataset --weights=last
```

Train with the pre-built model from this repository

```
python whale.py train --dataset=/path/to/dataset --weights=cetacean_photogram_model.h5
```