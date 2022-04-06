# Midterm Report (Machine Learning, CS7641)
#### Team 28: Aditya, Adwait, Saranya, Mihir, Tejas
#### Date: 5th April, 2022

________________________

## Introduction and Background

We will use the ongoing kaggle competition [H&M Personalized Fashion Recommendations](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/overview)’s dataset for this project. The topic of the project is **recommender systems**. The dataset contains purchase history of each customer along with additional metadata about the product (product group, description, image) and about the customer (age, zipcode). The goal of the project it to build a recommender system to predict with high accuracy, products that the customer may purchase in the future. Recommender systems is a widely studied topic with methods including collaborative methods, content based methods, knowledge based methods etc. We will primarily explore collaborative and content based methods in this project. We will rely on prior work [5, 6, 7] particularly for how RecSys has been approached for fashion recommendations

## Problem Definition and Motivation

Given the purchase history of customers along with metadata about both the product and the customer, our goal is to predict what products the customer will purchase in the time duration of 2 months right after the training data ends. The problem is a standard timeseries recommendation system problem with additional threads in NLP and CV (via product descriptions, and images)

Product recommendation is a very important problem in the e-commerce industry. Presenting customers with relevant recommendations not only makes for a good customer experience but also helps with the company’s revenue

## Data Collection

As mentioned, we retrieved our dataset from the kaggle competition “H&M Personalized Fashion Recommendations”, upon which we performed some preliminary data exploration and preprocessing. Below are some of the analyses we derived from our exploration. 

<img width="972" alt="line_graph_full" src="https://user-images.githubusercontent.com/28340555/161873630-306f3a21-be27-4992-b1f5-b84c8cf536e4.png">

The dataset consists of 3 csv files; articles, customers and transactions. The transactions file contains the list of purchases made by customers over a period of 2 years, with details such as the price of the articles, the customer unique id, the article unique id, and the date of purchase. To understand the general trend of these purchases, the above line graph was created, which plots the number of purchases made over the months, and their respective confidence intervals.
As can be seen, the data retrieved starts from September 2018 and continues on till December 2020. There seems to be an overall pattern in increasing purchases during the first half of the year, which peaks in June, and then follows a sharper decline during the second half. Between each year, the month to month trend seems to be almost identical, as depicted by the 2019 and 2020 graphs. 

Due to the huge amount of data, and computational limitations, we decided to sample a portion of the dataset for our purpose. We extracted all the transactions made during the year 2020, which we further split into training, test and validation datasets. Our testing and validation datasets were randomly sampled from the last 2 months of data, so that our model will try to predict what a customer is likely to purchase based on their previous history of purchases in the 7 months prior. To understand the trends in this sampled data, and the split between the training and test data, the line graph below was plotted. From this visualization it can be inferred that the purchases seem to stay within an approximate range, the only deviation from this occurring in June, in line with our analyses of the complete dataset.

<img width="976" alt="line_graph_train_test" src="https://user-images.githubusercontent.com/28340555/161873693-8c67d665-f53a-491c-84a4-283529c0cffb.png">

To aggregate the data month wise and reinforce our findings from the first two plots, we created a bar graph that accumulates the number of purchases made over the sampled months of January to September, 2020. The resulting graph verifies the trend we observed earlier, with there being a general increase in purchases from May to June, at which point there is a sharper decline for the latter half of the year. This decline mostly occurs in the month of July, with purchases in the following months of August and September being similar.

<img width="938" alt="bar_graph" src="https://user-images.githubusercontent.com/28340555/161873770-7598a7ee-e889-49bf-b7eb-2ec915d90e49.png">

While the above exploration was with respect to the time series data gained from the transactions file, we also performed some exploration on the customer and article files, to gain clarity on their respective features and purchases trends. Below is a scatter plot, which takes in features such as the Customer age, the product types purchased, and the prices of these products. For easier understanding of the visualization, one week's worth of data and only the top 5 product types were extracted to obtain the below plot.

<img width="925" alt="scatterplot" src="https://user-images.githubusercontent.com/28340555/161873840-e7ad7d8c-a4db-48d0-a38e-d39b4968334b.png">

A couple of things stand out from the above analysis. First off, it’s evident that the most popular products are categorized as Upper Body garments, with accessories being next. Most of the accessories purchases lie on the lower end of the price spectrum, implying that popular accessories are usually not the high-end ones. The prices stick within a range 0 to 0.2, with very few outliers beyond this, and similarly a bulk of the customers lie within the age range of 20-80 years, indicating that most of our recommendations would take this age range into account. The plot above helps to understand the general customer’s purchasing budget, and which articles they would prefer for which prices.



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
