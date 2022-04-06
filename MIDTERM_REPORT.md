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

<img width="800" alt="line_graph_full" src="https://user-images.githubusercontent.com/28340555/161873630-306f3a21-be27-4992-b1f5-b84c8cf536e4.png">

The dataset consists of 3 csv files; articles, customers and transactions. The transactions file contains the list of purchases made by customers over a period of 2 years, with details such as the price of the articles, the customer unique id, the article unique id, and the date of purchase. To understand the general trend of these purchases, the above line graph was created, which plots the number of purchases made over the months, and their respective confidence intervals.
As can be seen, the data retrieved starts from September 2018 and continues on till December 2020. There seems to be an overall pattern in increasing purchases during the first half of the year, which peaks in June, and then follows a sharper decline during the second half. Between each year, the month to month trend seems to be almost identical, as depicted by the 2019 and 2020 graphs.

Due to the huge amount of data, and computational limitations, we decided to sample a portion of the dataset for our purpose. We extracted all the transactions made during the year 2020, which we further split into training, test and validation datasets. Our testing and validation datasets were randomly sampled from the last 2 months of data, so that our model will try to predict what a customer is likely to purchase based on their previous history of purchases in the 7 months prior. To understand the trends in this sampled data, and the split between the training and test data, the line graph below was plotted. From this visualization it can be inferred that the purchases seem to stay within an approximate range, the only deviation from this occurring in June, in line with our analyses of the complete dataset.

<img width="800" alt="line_graph_train_test" src="https://user-images.githubusercontent.com/28340555/161873693-8c67d665-f53a-491c-84a4-283529c0cffb.png">

To aggregate the data month wise and reinforce our findings from the first two plots, we created a bar graph that accumulates the number of purchases made over the sampled months of January to September, 2020. The resulting graph verifies the trend we observed earlier, with there being a general increase in purchases from May to June, at which point there is a sharper decline for the latter half of the year. This decline mostly occurs in the month of July, with purchases in the following months of August and September being similar.

<img width="800" alt="bar_graph" src="https://user-images.githubusercontent.com/28340555/161873770-7598a7ee-e889-49bf-b7eb-2ec915d90e49.png">

While the above exploration was with respect to the time series data gained from the transactions file, we also performed some exploration on the customer and article files, to gain clarity on their respective features and purchases trends. Below is a scatter plot, which takes in features such as the Customer age, the product types purchased, and the prices of these products. For easier understanding of the visualization, one week's worth of data and only the top 5 product types were extracted to obtain the below plot.

<img width="800" alt="scatterplot" src="https://user-images.githubusercontent.com/28340555/161873840-e7ad7d8c-a4db-48d0-a38e-d39b4968334b.png">

A couple of things stand out from the above analysis. First off, it’s evident that the most popular products are categorized as Upper Body garments, with accessories being next. Most of the accessories purchases lie on the lower end of the price spectrum, implying that popular accessories are usually not the high-end ones. The prices stick within a range 0 to 0.2, with very few outliers beyond this, and similarly a bulk of the customers lie within the age range of 20-80 years, indicating that most of our recommendations would take this age range into account. The plot above helps to understand the general customer’s purchasing budget, and which articles they would prefer for which prices.

## Sampling the dataset and Split Creation

The full dataset consists of customer transactions from September 2018 to September 2020 containing a total of 1,371,980 customers, 105,542 products and 31,788,324 transactions.

Due to resource and time constraints, as part of this project we only use the transaction data of only 100k users and limit ourselves to purchases made in the year 2020. The resulting sampled dataset consists of 100,000 customers, 82,320 products, and 2,325,536 transactions.

Since the goal of the project is to determine future purchase patterns of any given customer. We splice the data in the time dimension to create our train, validation and test sets.

The train set is constructed using all transactions made from 1st January 2020 to 22nd July 2020. The remaining data (July to September) is split randomly into two equal parts as validation and test set. Specifically the validation and test sets contain transactions made from 23rd July 2020 to 22nd September 2020. Details of our train, val and test split are listed in the table below

| Split       | # Customers  | # Articles  | # Articles  | Time period            |
|------------ |------------- |------------ |------------ |----------------------- |
| Train       | 96589        | 77134       | 2147280     | 1/1/2020 - 7/22/2020   |
| Validation  | 25099        | 15900       | 89128       | 7/23/2020 - 9/22/2020  |
| Test        | 22919        | 18949       | 89128       | 7/23/2020 - 9/22/2020  |

Visualizations done as part of exploring the dataset have already been described in the above section

## Methods

### Collaborative Filtering (Method 1)

Collaborative Filtering makes predictions based on user similarity. Its underlying principle is that if user A and user B have similar tastes, then articles bought by user A will probably be bought by user B too and vice-versa.
We used the train splits for user data from above and joined them with the transactions completed for the time range considered.

#### Matrix Factorization
We form a matrix A that consists of users and the articles bought by those users. This matrix is sparse as the number of articles purchased by each user is a small subset of all the articles bought by all the users. We use SVD and use the top k (=10) sigma values to rebuild matrix A. This resultant matrix gives us the probability that user X will be interested in buying an article based on the articles bought by other users that have the same taste as user X.

### Content-Based Filtering (Method 2)
Content-based methods use article metadata previously bought by the users and recommend articles similar to those. Its underlying principle is that if user A bought article X then there is a higher probability that user X will be interested in buying other articles similar to article X. For example, if a user previously bought a striped shirt, the algorithm will recommend other striped shirts for the user to buy. We used all the product data and joined them with the transactions completed for the time range considered.

#### TF-IDF Vectorization with Cosine Similarity
In our first approach, we combined all the article metadata into a single descriptive text and ran Term Frequency - Inverse Document Frequency (TF-IDF) Vectorization on this single descriptive text. After doing that, we calculated the cosine similarity between the articles based on the vectorization to group the articles into similar products. This approach was giving us poor results and hence we decided to group articles in combination with other advanced techniques.

#### Clustering Product images using K-Means
In our second approach, we used K-means clustering to group images of products into different categories. Here, categories conceptually refer to the product types viz. shirts, pants, shoes, and so on. As we are working with image data, this also takes into consideration the patterns or designs on the products. For instance, products with flowers or stripes may be clustered together.

##### VGG-16 Architecture
We used a modified version of the VGG-16 architecture to extract features from images.

<img width="800" alt="VGG-16 Architecture" src="https://user-images.githubusercontent.com/53764708/161888807-fa636a34-689c-481a-a034-a6f81e750993.png">

The architecture of VGG16 is depicted above. VGG-16 is a CNN model proposed in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition” by K. Simonyan and A. Zisserman. VGG-16 achieved a top-5 test accuracy of 92.7% in ImageNet [8].

Our task here is not to perform image recognition but to extract features from images. Because of this, we only take the feature layer, average pooling layer, and one fully-connected layer that outputs a 4096-dimensional vector.  Our neural network architecture is as follows:

```
FeatureExtractor(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (pooling): AdaptiveAvgPool2d(output_size=(7, 7))
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc): Linear(in_features=25088, out_features=4096, bias=True)
)
```
When we pass an image from this neural network, the image is flattened i.e. converted into a one-dimensional vector. This one-dimensional vector is passed through the neural network and a 4096-dimensional vector encoding the features of the input image is returned as an output. We need to transform the images so that they are of the same size and match the input dimensions required by the model.

<img width="600" alt="Feature shape" src="https://user-images.githubusercontent.com/53764708/161889660-3ab3d47d-dd64-484e-944d-5ed1c4107ea0.png">

Here, we see the shape and the actual features for an example image.
Next, we run the K-means clustering algorithm with the image features generated in the previous step as input. We create multiple clustering models with clusters ranging from 15 to 50 i.e. min_num_clusters = 15 and max_num_clusters = 50. For each num_clusters k, we calculate the Silhouette score and Davies-Bouldin score in order to evaluate the goodness of clustering.

<img width="400" alt="Silhoutte" src="https://user-images.githubusercontent.com/53764708/161889800-b55d3d61-db69-4cad-bc12-0c5296c8c607.png">

We also use the Elbow method to determine the optimum number of clusters.

<img width="600" alt="Elbow Method" src="https://user-images.githubusercontent.com/53764708/161889908-da34e4fc-519b-4c0e-80d4-359da5674970.png">

We can use either of these scores to determine the optimal number of clusters. As an example, we use the Elbow method. As it can be seen in the above graph, we have an elbow at 34. So the optimal number of clusters is 34.

Next, we display the images corresponding to some clusters in order to understand and visualize how the clusters have been formed. Following are some examples of the same:

<img width="800" alt="Cluster A" src="https://user-images.githubusercontent.com/53764708/161890077-44c3c1f6-0512-451a-94c7-10e5920fab51.png">
a. This cluster mostly consists of jeans and trousers.

<br><br>

<img width="800" alt="Cluster B" src="https://user-images.githubusercontent.com/53764708/161890246-b59a8e93-11ec-4e54-9c80-869382b3105e.png">
b. This cluster contains products having ‘stripes’ design.

<br><br>

<img width="800" alt="Cluster C" src="https://user-images.githubusercontent.com/53764708/161890287-8363d7a8-2f1d-4c96-9fc2-1ce6841d3c0f.png">
c. This cluster consists of products with ‘floral’ design.

<br><br>

From these example clusters, we can see that products having similar design or style are clustered together. This can be used to recommend products to the customers based on their previous purchases. For example, if a person bought 3 dresses with floral patterns in the last month, we can recommend the floral dresses (same cluster) that are most similar to the ones they purchased earlier (intra-cluster distance) to them.

## Evaluation
We use the Mean Average Precision @k (i.e. MAP@k) metric to evaluate  the performance of our recommender system

### Mean Average Precision @ k (MAP@k)
Mean Average Precision(MAP@k) is a performance metric especially suited for recommender systems to evaluate the recommendations as well as the order of the recommendations. It's essentially the mean of Average Precision @ k (or AP@k) aggregated over all the users in the dataset. The pseudo code for AP@k is provided below

<img width="399" alt="Screen Shot 2022-04-06 at 12 52 27 AM" src="https://user-images.githubusercontent.com/7334811/161898325-32e970f3-cd40-4f8e-b77c-ed05a17157f8.png">

Where `actual` refers to the list of products that the user has already purchased (ground truth) and `predicted` (predictions) refers to the list of products recommended by our system, `in order of relevance` (i.e. order of the items in predicted matters)

 It is a slight modification of the AP metric with the salient difference being that AP@k takes into account the order of predictions i.e., it is not sufficient to recommend 10 items where the first 5 are irrelevant and the last 5 are highly relevant. Our recommender system should make sure that relevant products are predicted with high confidence. We provide the pseudo code for MAP@k below. In accordance with the [competition's guidelines](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview/evaluation), we use the MAP@12 metric as our main evaluation criterion

## Results and Discussion

Due to the computational requirement of our current method, we were not able to perform cross validation and then pick the best model. We hope to do a principled model selection / ensemble step as part of the final report.
### Quantitative Evaluation
We calculate the MAP@k metric on the validation and test splits and obtain the following performance. The table below shows the MAP@k values of our model on the validation and test set for varying k

| k  | Validation MAP@k | Test MAP@k |
|----|------------------|------------|
| 10 | 0.004830         | 0.004850   |
| 12 | 0.004851         | 0.004856   |
| 20 | 0.004963         | 0.00495680 |
| 25 | 0.005021         | 0.0050513  |
| 50 | 0.0055747        | 0.0053765  |

Below you can find the histograms of our AP@50 scores. Notice that such low values are actually expected for a recommendation system especially for product recommendations. For reference, the top performing model in the kaggle leaderboard right now (which is a mixture of SOTA methods ensembled and trained on the entire dataset of 2 years) is 0.03 (Screenshot attached below for your reference)

<img width="950" alt="Screen Shot 2022-04-06 at 1 24 46 AM" src="https://user-images.githubusercontent.com/7334811/161901691-a6c96d79-1ece-4a86-be29-fb0f08e216f6.png">

#### Histogram of AP@50 scores for all of the customers in the test set
<img width="401" alt="Screen Shot 2022-04-06 at 1 23 12 AM" src="https://user-images.githubusercontent.com/7334811/161901518-1b61a873-4eb2-4745-a016-0e1e334f9ac9.png">

#### Histogram of AP@50 scores by neglecting scores less than 0.01

<img width="399" alt="Screen Shot 2022-04-06 at 1 23 50 AM" src="https://user-images.githubusercontent.com/7334811/161901587-fb334278-6228-4952-b457-5b2a015e1216.png">

### Qualitative Evaluation

## Potential Results and Discussion

We hope to have![Uploading Screen Shot 2022-04-05 at 11.21.29 PM.png…]()
e the following results at the end of the semester with 1. and 2. being ready for the mid-term report

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
8. Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
