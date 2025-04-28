# Project Description
This is my project of DSCI-498 & DSCI-441.

My project based on EarthVQA, a paper in ResearchGate, which is mainly focus on remote sensing and visual question & answer. Here are the details about the the [Discription of EarthVQA](ori_README.md)


## Project Title
Deep Vision Models for Remote Sensing Applications

For DSCI-498, it is a deep learning based VQA model
For DSCI-441, it is a machine learning based classification model

## Projects Abstract
Last semester, all of my projects are about climate change. As I wrote in the descrpition, I am very interested in the combination between machine learning and atmosphere, geology, earth science. So, this semester I decide to do something about these topics.

This project mainly consists 2 parts. 
- Remote sensing segmentation
- Vision based Q&A

The first part will use an generative mode, which generate a segmentation output (in original EarthVQA, it use a FPN. I am not sure what I will use in the final project). 

The second part will combine the visual segmentation features and text features, and generate the text output as answer. This part is a discriminative model, but I am finding the way to replace it by a generative model like GPT, that would be much interesting.

Besides, I also want to check my understanding for those traditional machine learning model, I did several experiments on the almost same dataset.

## Dataset
- [EarthVQA Dataset](https://forms.office.com/r/g6hr92aCj5) is my first choice for VQA task
- [LoveDA Dataset](https://zenodo.org/records/5706578) is my first choice for classification task
- Also, some related dataset:
    - [RS-CQMA (Global-TQA)](https://github.com/shenyedepisa/RSCMQA?tab=readme-ov-file)
    - [EarthMaker](https://github.com/wivizhang/EarthMarker)
    - [ISPRS](https://www.isprs.org/resources/datasets/images/)


## How to run the code

### For DSCI-498 "A VQA model of remote sensing field"
- Install the environment
```
pip install ever-beta
pip install git+https://github.com/qubvel/segmentation_models.pytorch
pip install albumentations==1.4.3 # This version is important for our repo.
```
or just install by `requirement.txt`
```
pip install -r #Classification_task/requirements.txt
```
- Run the "EarthVQA Demo"
```
streamlit run \#EarthVQA_demo/earthvqa_app.py --server.port 8501 --server.headless true
```



### For DSCI-441 "A Classification model of remote sensing field"
- Install the environment
```
pip install -r \#Classification_task/requirements.txt
```

- Run the "classification model"
```
streamlit run \#Classification_task/streamlit_classification_app.py --server.port 8501 --server.headless true
```

- Also if you want to test/train the model by hand:
```
python \#Classification_task/rf_main.py
python \#Classification_task/cnn_main.py
python \#Classification_task/xgboost_main.py
python \#Classification_task/_main.py
```