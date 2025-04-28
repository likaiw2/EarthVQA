### Data preparation
- [Download EarthVQA dataset and pre-trained weights](https://github.com/Junjue-Wang/EarthVQA)
- Construct the data as follows:
```none
EarthVQA
├── Train
│   ├── images_png
│   ├── masks_png
├── Val
│   ├── images_png
│   ├── masks_png
├── Test
│   ├── images_png
├── Train_QA.json
├── Val_QA.json
├── Test_QA.json
├── log
|   |—— sfpnr50.pth
│   ├── soba.pth
```
Note that the images are the same as the LoveDA dataset, so the urban and rural areas can be divided on your own.