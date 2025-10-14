## One-Hot Encoding in Machine Learning

# Overview:-
-> This project focuses on "One-Hot Encoding (OHE)" — a key preprocessing step used to convert categorical variables into a numerical format suitable for machine learning algorithms.  
-> Here, we explore multiple ways to implement OHE using "Pandas" and "Scikit-learn", along with variations like "(k−1) encoding" and "encoding top categories only".


# Objectives:-
-> Understand the concept and importance of One-Hot Encoding  
-> Implement OHE using `pandas.get_dummies()`  
-> Perform (k−1) encoding to avoid multicollinearity  
-> Apply OHE using `sklearn.preprocessing.OneHotEncoder`  
-> Encode only 'top categories' to handle high-cardinality features  


# Techniques / Concepts Used:-
-> `pandas.get_dummies()`  
-> (k−1) Encoding (drop first category)  
-> `OneHotEncoder` from `sklearn.preprocessing`  
-> Handling high-cardinality categorical features using `top_categories` approach  
-> Data visualization using **Matplotlib** and **Seaborn**


# Dataset:-
-> The dataset contains categorical features representing various attributes (e.g., city, department, category).  
-> These columns are encoded into binary dummy variables to make them suitable for model input.


# Implementation Steps:-
1. Import required libraries (`numpy`, `pandas`, `sklearn`)  
2. Load and explore dataset  
3. Apply One-Hot Encoding using:
   -> "Pandas (`get_dummies`)"
   -> "Pandas with (k−1) Encoding"
   -> "Scikit-learn’s OneHotEncoder"
4. Handle **top categories** using frequency-based filtering  
5. Compare and visualize results  


# Key Observations:-
-> OHE creates new binary columns for each unique category.  
-> (k−1) encoding prevents redundancy and multicollinearity in linear models.  
-> Scikit-learn’s encoder provides more flexibility (e.g., `handle_unknown='ignore'`).  
-> Limiting to **top categories** helps when there are too many unique values.  


# Conclusion:-
-> One-Hot Encoding is one of the most essential steps in feature preprocessing.  
-> Choosing between Pandas and Scikit-learn implementations depends on use case complexity — simple datasets work well with Pandas, while pipelines and model training benefit from 
    Scikit-learn’s OHE.


# Technologies Used:-
-> Python 🐍  
-> NumPy  
-> Pandas  
-> Scikit-learn  
-> Matplotlib  
-> Seaborn  
