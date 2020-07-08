# Random Forest Regression on the World Population

**Completed by Mangaliso Makhoba.**

**Overview:** Use ensamble methods, Random Forest Regressor, to predict the population of a region at an input year, according to the population's income group. 

**Problem Statement:** Predict population at a ___year___, given the ___income group___

**Data:** [World Population Dataset (1960 - 2017)](https://raw.githubusercontent.com/Explore-AI/Public-Data/master/AnalyseProject/world_population.csv)

**Deliverables:** A predictive Model

## Topics Covered

1. Machine Learning
3. Ensamble Methods
4. Raandom Forest Regression
5. Mean Squared Error
6. Kfold

## Tools Used
1. Python
1. Pandas
2. Scikit-learn
2. Jupyter Notebook

## Installation and Usage

Ensure that the following packages have been installed and imported.

```bash
pip install numpy
pip install pandas
pip install sklearn
```

#### Jupyter Notebook - to run ipython notebook (.ipynb) project file
Follow instruction on https://docs.anaconda.com/anaconda/install/ to install Anaconda with Jupyter. 

Alternatively:
VS Code can render Jupyter Notebooks

## Notebook Structure
The structure of this notebook is as follows:

 - First, we'll load our data to get a view of the predictor and response variables we will be modeling. 
 - Then get the total population of the world by income group. 
 - We then split the data into K-fold train and test datasets.
 - Following this modeling, we define a custom metric as the log-loss in order to evaluate our produced model.
 - Train the model with each split and return the trained model with lowest Mean Squared Error 



# Function 1: Income Group Selection
Write a function that takes as input an income group and return a 2-d numpy array that contains the year and the measured population.

_**Function Specifications:**_
* Should take a `str` argument, called `income_group_name` as input and return a numpy `array` type as output.
* Set the default argument of `income_group_name` to equal `'Low income'`.
* If the specified value of `income_group_name` does not exist, the function must raise a `ValueError`.
* The array should only have two columns containing the year and the population, in other words, it should have a shape `(?, 2)` where `?` is the length of the data.
* The values within the array should be of type `np.int64`. 

_**Further Reading:**_

Data types are associated with memory allocation. As such, your choice of data type affects the precision of computations in your program. For example, the `np.int` data type in numpy can only store values between -2147483648 to 2147483647 and assigning values outside this range for variables of this data type may cause run-time errors. To avoid this, we can use data types with larger memory capacity e.g. `np.int64`.


_**Expected Outputs:**_
```python
get_total_pop_by_income('High income')
```
```python
array([[      1960,  769889923],
       [      1961,  781225329],
       [      1962,  791207437],
       [      1963,  801108277],
       ...
       [      2015, 1211252041],
       [      2016, 1218629612],
       [      2017, 1225514228]])
```



# Function 2: Model Training

Now that we have have our data, we need to split this into a set of variables we will be training on, and the set of variables that we will make our predictions on.

Unlike in the previous coding challenge, a friend of ours has indicated that sklearn has a bunch of built-in functionality for creating training and testing sets. In this case however, they have asked us to implement a k-fold cross validation split of the data using sklearn's `KFold` [class](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) (which has already been imported into this notebook for your convenience). 

Using this knowledge, write a function which uses sklearn's `KFold` [class](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) internally, and that will take as input a 2-d numpy array and an integer `K` corresponding to the number of splits. This function will then return a list of tuples of length `K`. Each tuple in this list should consist of a `train_indices` list and a `test_indices` list containing the training/testing data point indices for that particular Kth split.

_**Function Specifications:**_
* Should take a 2-d numpy `array` and an integer `K` as input.
* Should use sklearn's `KFold` [class](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html).
* Should return a list of `K` `tuples` containing a list of training and testing indices corresponding to the data points that belong to a particular split. For example, given an array called `data` and an integer `K`, the function should return: 
```python
data_indices = [(list_of_train_indices_for_split_1, list_of_test_indices_for_split_1)
                  (list_of_train_indices_for_split_2, list_of_test_indices_for_split_2)
                  (list_of_train_indices_for_split_3, list_of_test_indices_for_split_3)
                                                   ...
                                                   ...
                  (list_of_train_indices_for_split_K, list_of_test_indices_for_split_K)]
```

* The `shuffle` argument in the KFold object should be set to `False`.

_**Expected Outputs:**_
```python
data = get_total_pop_by_income('High income')
sklearn_kfold_split(data,4)
```
```python
[(array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
         49, 50, 51, 52, 53, 54, 55, 56, 57]),
  array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])),
 (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 30, 31,
         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
         49, 50, 51, 52, 53, 54, 55, 56, 57]),
  array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])),
 (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 44, 45, 46, 47,
         48, 49, 50, 51, 52, 53, 54, 55, 56, 57]),
  array([30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43])),
 (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
         34, 35, 36, 37, 38, 39, 40, 41, 42, 43]),
  array([44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]))]
 ```

# Function 3: Select Best Performing Split Model

Now that we have formatted our data, we can fit a model using sklearn's `RandomForestRegressor` class. We'll write a function that will take as input the data indices (consisting of the train and test indices for each split) that we created in the last question, train a different `RandomForestRegressor` on each split and return the model that obtains the best testing set performance across all K splits.

**Important Note:** Due to the random initialisation process used within sklearn's `RandomForestRegressor` class, you will need to fix the value of the `random_state` argument in order to get repeatable and predictable results.

_**Function Specifications:**_
* Should take a 2-d numpy array (i.e. the data) and `data_indices` (a list of `(train_indices,test_indices)` tuples) as input.
* For each `(train_indices,test_indices)` tuple in `data_indices` the function should:
    * Train a new `RandomForestRegressor` model on the portion of data indexed by `train_indices`
    * Evaluate the trained `RandomForestRegressor` model on the portion of data indexed by `test_indices` using the **mean squared error** (which has also been imported for your convenience).
* After training and evalating the `RandomForestRegressor` models, the function should return the `RandomForestRegressor` model that obtained highest testing set `mean_square_error` over its allocated data split across all trained models. 
* The trained `RandomForestRegressor` models should be trained with `random_state` equal `42`, all other parameters should be left as default.

For each tuple in the `data_indices` list, you can obtain `X_train`,`X_test`, `y_train`, `y_test` as follows:  
```python
    X_train, y_train = data[train_indices,0],data[train_indices,1]
    X_test, y_test = data[test_indices,0],data[test_indices,1]
```
_**Expected Outputs:**_
```python
data = get_total_pop_by_income('High income')

data_indices = sklearn_kfold_split(data,5)

best_model = best_k_model(data,data_indices)

best_model.predict([[1960]]) == array([8.85170916e+08])
```

## Conclusion
It is useful to split large data and train it in chunks provided the data normally follows a certain trend such as population data. This will help avoid outliers provided the sample chosen is **NOT** itself the "outlier", it would really be odd; in which case we will have to consider the contextual events which could have lead to any abnomalities in the Data. To overcome this, . However, Ensamble Methods, such as the Random Forest, are generally aggressive to outliers, therefore, outliers would not heavily deviate the model's perfomance. 

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. 

## Contributing Authors
**Authors:** Mangaliso Makhoba, Explore Data Science Academy

**Contact:** makhoba808@gmail.com

## Project Continuity
This is project is complete


## License
[MIT](https://choosealicense.com/licenses/mit/)