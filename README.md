# Project Description
This is my project of DSCI-498 & DSCI-441.

My project based on EarthVQA, a paper in ResearchGate, which is mainly focus on remote sensing and visual question & answer. Here are the details about the the [Discription of EarthVQA](ori_README.md)

Here are the summary of my project.

## Project Title
Anvanced EarthVQA

## Project Abstract
Last semester, all of my projects are about climate change. As I wrote in the descrpition, I am very interested in the combination between machine learning and atmosphere, geology, earth science. So, this semester I decide to do something about these topics.

This project mainly consists 2 parts. 
- Remote sensing segmentation
- Vision based Q&A

The first part will use an generative mode, which generate a segmentation output (in original EarthVQA, it use a FPN. I am not sure what I will use in the final project). 

The second part will combine the visual segmentation features and text features, and generate the text output as answer. This part is a discriminative model, but I am finding the way to replace it by a generative model like GPT, that would be much interesting.

## Dataset
- [EarthVQA Dataset](https://forms.office.com/r/g6hr92aCj5) is my first choice
- Also, some related dataset:
    - [RS-CQMA (Global-TQA)](https://github.com/shenyedepisa/RSCMQA?tab=readme-ov-file)
    - [EarthMaker](https://github.com/wivizhang/EarthMarker)
