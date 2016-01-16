# spatio-time-series-clustering-
machine learning project based on git alexminnaar

In this project , every stock has two values: 1) time invariant called features 2)time variant called return of a stock. And the time variant made up the time series.</br>
Time series and features have different characteristics and represent different identification of a system. If we want to cluster the stocks we should both measure the time series and features. This is generally called Spatio-temporal clustering method.</br>
So in this part, we use the cluster method to forecast the return of the stocks in the test set.Here are the experimental steps:
Firstly, we used the spatio-temporal clustering technique similar to K-means clustering to classify different stocks in training set into the groups. And finding the central time series like the centroids in k-means which represent the common pattern for each group series.</br>
Secondly, we predict the test stocks belongs to which groups. Then we use the central time series as the future return to forecast. </br>
Finally , we use the evaluation function as our performance criterion</br>


