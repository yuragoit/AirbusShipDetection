# [Semantic Segmentation (Airbus Ship Detection Challenge)]

You can go to (28.07.23 planned deployment of a Semantic Segmentation module to main App):
<a href='https://webmlassistantteam2-production.up.railway.app/' target="_blank">LIVE Web-ML Assistant App</a>

# Stack of technologies used
<a href="https://www.tensorflow.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="30" height="30"/> </a>
<a href="https://keras.io/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Keras_logo.svg/512px-Keras_logo.svg.png?20200317115153" alt="Keras" width="30" height="30"/> </a>
<a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="Scikit-learn" width="30" height="30"/> </a>
<a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="30" height="30"/> </a> 
<a href="https://www.djangoproject.com" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/djangoproject/djangoproject-icon.svg" alt="django" width="30" height="30"/> </a> 
<a href="https://www.w3.org/html/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/html5/html5-original-wordmark.svg" alt="html5" width="30" height="30"/> </a> 
<a href="https://www.w3schools.com/css/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/css3/css3-original-wordmark.svg" alt="css3" width="30" height="30"/> </a>
<a href="https://www.w3schools.com/js/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/javascript/javascript-icon.svg" alt="js" width="30" height="30"/> </a>
<a href="https://www.postgresql.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/postgresql/postgresql-icon.svg" alt="postgresql" width="30" height="30"/> </a>
<a href="http://nginx.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/nginx/nginx-icon.svg" alt="nginx" width="30" height="30"/> </a>
<a href="https://gunicorn.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/gunicorn/gunicorn-icon.svg" alt="gunicorn" width="30" height="30"/> </a>
<a href="https://git-scm.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" width="30" height="30"/> </a>
<a href="https://www.w3schools.io/terminal/bash-tutorials/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/gnu_bash/gnu_bash-icon.svg" alt="bash" width="30" height="30"/> </a>
<a href="https://www.linux.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/linux/linux-icon.svg" alt="linux" width="30" height="30"/> </a>
<a href="https://getbootstrap.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/getbootstrap/getbootstrap-icon.svg" alt="linux" width="30" height="30"/> </a>

# About the project

- Main idea of the project: creating pipeline using tf.keras for training U-Net model to solve semantic segmentation task from Airbus Ship Detection Challenge
- Deadline: MVP - 3 days, Prod - 1 day
- Supported modules: Computer Vision (Semantic Segmentation)
- Dataset based on [ASD-dataset](https://www.kaggle.com/c/airbus-ship-detection/data)

# Manual Build / Access

> Download the code 

```bash
$ # Get the code
$ git clone https://github.com/yuragoit/AirbusShipDetection.git
$ cd AirbusShipDetection
```

### Set Up for Unix, MacOS

> Install modules via venv  

```bash
$ virtualenv env
$ source env/bin/activate
$ pip3 install -r requirements.txt
```

> Start train / test process

```bash
$ python train.py
$ python inference.py
```

### Set Up for Windows

> Install modules via venv (Windows) 

```
$ virtualenv env
$ .\env\Scripts\activate
$ pip3 install -r requirements.txt
```

> Start train / test process

```bash
$ python train.py
$ python inference.py
```

## Description of solution
### Preparation

* a. Create a base directory (e.g. 'airbus-ship-detection') with two subfolders 'train_v2' and 'test_v2'.
* b. Download dataset from Kaggle: [Airbus Ship Detection Challenge](https://www.kaggle.com/competitions/airbus-ship-detection/data). 
* c. Unzip data two appropriate subfolders according to their names ('train_v2' and 'test_v2').
* d. Check project / base dir structure with dataset:
<pre>
 ├── airbus-ship-detection/
 |    ├── train_v2/*.jpg
 |    ├── test_v2/*.jpg
 |    ├── train_ship_segmentations_v2.csv
 |    ├── sample_submission_v2.csv
 ├── models/
 |    ├── seg_unet_model.h5
 |    └── seg_model_weights.best.hdf5
 ├── notebooks/
 |   ├── asd-unet-keras-dice-v9.ipynb
 ├── README.md
 ├── inference.py
 ├── train.py
 └── requirements.txt
</pre>

### Files description:

1. train.py - file contains pipeline for training U-Net model to solve semantic segmentation task.
2. inference.py - file for results visualization / testing predictions.
3. seg_unet_model.h5 - fill resolution (768x768) pretrained Unet model.
4. asd-unet-keras-dice-v9.ipynb - EDA with train/test pipeline.

### EDA with test / train pipeline
EDA with test / train pipeline defined in **asd-unet-keras-dice-v9.ipynb** file.
Unbalanced / balanced ship distribution shown below

![image](https://github.com/yuragoit/webMLAssistantTeam2/assets/101989870/915a58dc-83ba-4907-af23-01f4e81b06ec)


### Architecture:

 - Architecture: U-Net
 - Loss function: Combo_loss (mixing binary_crossentropy with dice_coef)
 - Optimizer: Adam (learning_rate=1e-4, weight_decay=1e-6)
 - Learning scheduler with EarlyStopping: ReduceLROnPlateau(factor=0.5, patience=3)
 
### Inference / Results
 
The best results have been obtained with Combo_loss for Semantic Segmentation task.

| Architecture | val_binary_accuracy | val_loss  | val_dice_coef |
|--------------|---------------------|-----------|---------------|
| U-net        | 0.999               | 0.0291    | 0.7784        |

Prediction mask 1:

![image](https://github.com/yuragoit/webMLAssistantTeam2/assets/101989870/f29a3275-3b9f-41a0-b10e-db72588f8dff)
 
Prediction mask 2: 

![image](https://github.com/yuragoit/webMLAssistantTeam2/assets/101989870/3d0a0750-872a-43aa-99e7-d90cfda90c24)
 
Prediction mask 3: 

![image](https://github.com/yuragoit/webMLAssistantTeam2/assets/101989870/83e419d4-cc47-4db8-a41a-91b896d650d4)

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Used By

This project is used by the following companies:

- LLC WT


