# Hierarchical Modeling for Task-Recognition and Action Segmentation in Weakly-Labeled Instructional Videos


Main Software Requirements:
PyTorch 1.1
Python 3.6
numpy among others
Ubuntu 16





********************************************************************************************************************
Instructions to reproduce the task recognition results on the Beakfast dataset using I3D and iDT features:
********************************************************************************************************************
## Make sure you have the following folder directories

|data

	| i3d
	
		| features
	
	| idt 
	
		| features

	| groundTruth

	| transcripts
	
|logs

|Visualization

|utils_folder 

## Data Preparation
0-0- Download the pre-computed I3D features from the third party link used in [1]:  https://zenodo.org/record/3625992#.X7vj8axKjCJ

0-1- Extract the content of the "/breakfast/features/"  inside the defined "/data/i3d/features/" directory.

0-2- Download the pre-computed iDT from the third party link used in [2]: https://uni-bonn.sciebo.de/s/wOxTiWe5kfeY4Vd

0-3- Extract the content of the "data/features/"  inside our defined "/data/idt/features/" directory.

0-4- Extract the content of the "data/groundTruth/"  inside our defined "/data/groundTruth/" directory. (already done)

0-5- Extract the content of the "data/transcripts/"  inside our defined "/data/transcripts/" directory. (already done)

0-6- Place all the .py files in the same directory as data

## Execution

1-0- Go to options.py and change the parameters if desired. 

1-1- Type the following command line in terminal: python main.py

###############################
  


 





[1] Y. Abu Farha and J. Gall.
MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation.
In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019

[2] A. Richard, H. Kuehne, A. Iqbal, J. Gall:
NeuralNetwork-Viterbi: A Framework for Weakly Supervised Video Learning
in IEEE Int. Conf. on Computer Vision and Pattern Recognition, 2018
