## Marin mammal re-identification project 
2022년에 해양 포유류 재확인 (Marine mammal re-identification) 프로젝트를 수행했습니다.



## 활용 기술
```
Python  
CNN(Convolutional Neural Network)
Segmentation
Pytorch
```

## About task
- 고래 및 돌고래 등의 개체 사진 활용
- 새롭게 찍힌 사진에 나타나는 개체가 기존에 수집했던 자료 중 어떤 개체와 일치하는지 판정

## 1. Dataset
[해양 포유류 데이터](https://www.kaggle.com/competitions/happy-whale-and-dolphin/)를 활용하여 인공지능 모델 학습

## 2. Data preprocessing
- [CLIP](http://proceedings.mlr.press/v139/radford21a/radford21a.pdf)을 이용한 image segmentation([Image Segmentation Using Text and Image Prompts, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Luddecke_Image_Segmentation_Using_Text_and_Image_Prompts_CVPR_2022_paper.pdf)) 기술 사용
- Text prompt로 "dolphin" 사용 <br/>

<img src = "./figures/clip_seg.PNG" width="65%"><br/><br/>

- Original image를 resize하여 W,H가 같은 image 생성
- 해당 image에 segmentation 수행하여 min, max 좌표 획득
- Resize했던 비율을 역이용하여 original image의 bounding box 좌표 획득

<img src = "./figures/clip_seg2.PNG" width="80%"><br/><br/>

- Original image의 객체와 배경의 비율을 맞추기 위해 zero padding을 사용하여 crop을 하거나 image를 붙여 crop
- 인공지능 모델의 입력 size를 맞추기 위해 resize 수행

<img src = "./figures/clip_seg3.PNG" width="80%"><br/><br/>

## 3. Re-identification & performance evaluation

3-1 Overall framework

- 총 4단계로 구성

<img src = "./figures/Overall framework.PNG" width="80%"><br/><br/>

3-2 Overall framework

- 

<img src = "./figures/Network structure.PNG" width="80%"><br/><br/>

<img src = "./figures/Make Gallery.PNG" width="80%"><br/><br/>

<img src = "./figures/Make Gallery2.PNG" width="80%"><br/><br/>

<img src = "./figures/Performance Evaluation.PNG" width="80%"><br/><br/>

<img src = "./figures/Performance Evaluation2.PNG" width="80%"><br/><br/>

<img src = "./figures/Performance Evaluation3.PNG" width="80%"><br/><br/>

<img src = "./figures/Performance Evaluation4.PNG" width="80%"><br/><br/>

## 4. Conclusion

- aa


<img src = "./figures/Test for Kaggle submission.PNG" width="80%">

















