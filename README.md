# Project Proposal (Machine Learning, CS7641)
#### Team 28: Aditya, Adwait, Saranya, Mihir, Tejas

________________________

## Introduction and Background

We will use the ongoing kaggle competition [H&M Personalized Fashion Recommendations](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/overview)’s dataset for this project. The topic of the project is **recommender systems**. The dataset contains purchase history of each customer along with additional metadata about the product (product group, description, image) and about the customer (age, zipcode). The goal of the project it to build a recommender system to predict with high accuracy, products that the customer may purchase in the future. Recommender systems is a widely studied topic with methods including collaborative methods, content based methods, knowledge based methods etc. We will primarily explore collaborative and content based methods in this project. We will rely on prior work [5, 6, 7] particularly for how RecSys has been approached for fashion recommendations

## Problem Definition and Motivation

Given the purchase history of customers along with metadata about both the product and the customer, our goal is to predict what products the customer will purchase in the time duration of 7 days right after the training data ends. The problem is a standard timeseries recommendation system problem with additional threads in NLP and CV (via product descriptions, and images)

Product recommendation is a very important problem in the e-commerce industry. Presenting customers with relevant recommendations not only makes for a good customer experience but also helps with the company’s revenue

## Methods

Since this is not a standard academic dataset and is an ongoing competition, we will be implementing the baseline approaches ourselves. Details of our plan are provided below

1. **Collaborative Filtering**[1]: We will use user-product interactions without taking into consideration the content of the product
2. **Content based methods**[2]: Unlike 1, in content based methods we will use product descriptions (text), product images (image) information to improve our recommendations
3. **Content based methods using neural networks**: We will improve upon 2. by using neural network based features instead of classical features


## Potential Results and Discussion

We hope to have the following results at the end of the semester with 1. and 2. being ready for the mid-term report

1. **Exploratory data analysis**: Going over the dataset, understanding the most important features for the task, combining existing features, data visualization

2. **Matrix Factorization**: We hope to have our final results for the matrix factorization method for the mid-term report. We will start with standard matrix factorization methods proposed in [1], and modify the method according to our use case

3. **Content based methods**[3]: Since our dataset contains a lot of information about the products themselves (product group, image, product description) it is fair to assume that using this additional data would help boost the performance obtained via the matrix factorization method. To incorporate content information, we will use classical vision and nlp models such as sift features, bag of words model

4. **Neural network based methods**[4]: Since neural networks have proven time and again to be powerful feature extractors, in this stage of the project we will rely on pretrained vision and language models to generate features for the product images and descriptions and use these features in our recommendation systems pipeline.

We hope to have the following results as part of the final report
1.  Comparative evaluation of the 3 methods: We believe that Neural network based methods will perform the best whereas Matrix factorization will perform the worst of our chosen methods
2.  Ablation study measuring the importance of different features

## Proposed Timeline

| Milestone | Completion date |
|-----------|------|
|Exploratory data analysis, cleaning and visualization | February 27th|
|Feature Reduction/selection | February 27th|
|Content based methods | March 6th|
|Matrix factorization -| March 14th|
|Coding and Implementation | March 17th|
|Midterm Report | April 5th|
|Neural Network model implementation | April 20th|
|Evaluation and Comparison of models | April 26th|
|Final Report | April 26th|

Our detailed timeline can be found [here](https://docs.google.com/spreadsheets/d/1x-xW91rFzp30riCjQ-ZyBcnbJ8fQewvw3XGVG49_Lqw/edit?usp=sharing)

## Proposed Contributions

|Member name | Task |
|------------|------|
| Aditya     | Feature selection and Neural network based methods (method 3) |
| Saranya    | EDA and Matrix Factorization (method 1) |
| Tejas      | Data visualization and Content based methods (method 2) |
| Adwait     | Literature review and Pretrained NLP models for neural network |
| Mihir      | Data cleaning and implementation of method 3 (NLP features) |
| All        | Evaluation of all methods, debugging, mid-term and end-term reports |

All team members will actively participate in discussions via a private slack workspace and keep each other in the loop

## Proposal Video

Video link: [https://www.youtube.com/watch?v=hyNbVMK_bNY](https://www.youtube.com/watch?v=hyNbVMK_bNY)

## References

1. Gábor Takács et al (2008). Matrix factorization and neighbor based algorithms for the Netflix prize problem. In: Proceedings of the 2008 ACM Conference on Recommender Systems, Lausanne, Switzerland, October 23 - 25, 267-274.
2. Lowe, David G. (1999). "Object recognition from local scale-invariant features" (PDF). Proceedings of the International Conference on Computer Vision. Vol. 2. pp. 1150–1157. doi:10.1109/ICCV.1999.790410.
3. Bag of words model., Chapter 6, Foundations of Statistical Natural Language Processing, 1999.
4. Ian Goodfellow, Yoshua Bengio, & Aaron Courville (2016). Deep Learning. MIT Press.
5. Naumov, Maxim & Mudigere, Dheevatsa & Shi, Hao-Jun & Huang, Jianyu & Sundaraman, Narayanan & Park, Jongsoo & Wang, Xiaodong & Gupta, Udit & Wu, Carole-Jean & Azzolini, Alisson & Dzhulgakov, Dmytro & Mallevich, Andrey & Cherniavskii, Ilia & Lu, Yinghai & Krishnamoorthi, Raghuraman & Yu, Ansha & Kondratenko, Volodymyr & Pereira, Stephanie & Chen, Xianjie & Smelyanskiy, Misha. (2019). Deep Learning Recommendation Model for Personalization and Recommendation Systems.
6. Yang Hu, Xi Yi, and Larry S. Davis. 2015. Collaborative Fashion Recommendation: A Functional Tensor Factorization Approach. In Proceedings of the 23rd ACM international conference on Multimedia (MM '15). Association for Computing Machinery, New York, NY, USA, 129–138. DOI:https://doi.org/10.1145/2733373.2806239
7. S. Liu, J. Feng, Z. Song, T. Zhang, H. Lu, C. Xu, and S. Yan. “Hi, magic closet, tell me what to wear!”. In ACM Multimedia, 2012.

