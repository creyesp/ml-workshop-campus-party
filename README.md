[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/creyesp/Meetup_uy/master)
[![Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667
)](https://colab.research.google.com/github/creyesp/Meetup_uy/blob/master/notebooks/Introduction_to_Scikit-Learn.ipynb?authuser=1)


# ml-workshop-campus-party

This project aims is explore the principal ideas behind of scikit-learn library. [The notebook](notebooks/Introduction_to_Scikit-Learn.ipynb) contain some simple example code to show how scikit-learn work, then we use a real dataset of Montevideo housing prices to create a real model to predict the price of each house.

This project was developed to be shared in [Montevideo Applied Data Science, AI and Big Data MeetUp](https://www.meetup.com/Montevideo-Applied-Data-Science-and-Big-Data/)

# Quick Start

## Look at notebook
If you can only look at notebook without run the code. Try [here](https://nbviewer.jupyter.org/github/creyesp/Meetup_uy/blob/master/notebooks/Introduction_to_Scikit-Learn.ipynb)

## Run in MyBinder server
MyBinder is a free service to run a repository with notebooks in a live sesion without install anything, just click [here](https://mybinder.org/v2/gh/creyesp/Meetup_uy/master).

**Warning**: MyBinder run a temporal sesion, when you leave sesion all your changes are going to be delete.

## Run in Google Colab
Colab is a google service to run notebook, but you can only run 1 notebook and to access to complete resources of the repository you should clone repository manually. 

Click [here](https://colab.research.google.com/github/creyesp/Meetup_uy/blob/master/notebooks/Introduction_to_Scikit-Learn.ipynb?authuser=1) to run in colab. 

When the sesion are ready run in a new cell the following commnads:

	!git clone https://github.com/creyesp/Meetup_uy.git
	!mv Meetup_uy/* .

Finally change all path to the current location, for example '../data/ready/properties.csv' to 'data/ready/properties.csv'

If you can save you changes you can save the notebook in your drive.

## Run in a local environment

Install python 3.5+ in your own machine and clone this repository following next step:

	$ git clone https://github.com/creyesp/Meetup_uy.git
	$ cd Meetup_uy
	$ virtualenv --python=python3.6 venv
	$ source venv/bin/activate
	$ python3 -m pip install --user --upgrade -r binder/requirements.txt
	$ jupyter notebook

