# Project for Big Data Mining and Analysis Course
### Movie Rating Prediction Project

- Dataset 
  - Download from http://files.grouplens.org/datasets/ movielens/ml-1m.zip.
  - The data set contains 1 million ratings from 6000 users on 4000 movies.
  - We further sort each ratings by timestamp
  
- Recommendations:
  - Step 1:
    - Baseline estimator: use the formular  bxi = μ + bx + bi on pdf
  - Step 2:
    - Neighborhood estimator: use the neighborhood approach to predict rating score
      - item-based similarity
      - user-based similarity
  - Incorporating Temporal Dynamics
    - Following the KDD09 Paper http://dl.acm.org/citation.cfm?doid=1721654.1721677
    - Equation 5,6,8,10
    
- K-mean Clustering 
  - Use the k-mean algorithm to cluster the users based on their rating scores given by the file ratings.dat.
  
- SVD Dimensionality Reduction
  - Use the SVD algorithm to reduce the dimensionality
  
- Metrics
  - the value of RMSE
  
- Project Implementation
  - Please refer to https://github.com/likicode/datamining_course_proj/blob/master/数据挖掘文档.pdf 
  - Recommendations part is coded with Python, the others are coded with Matlab by my group member. 
      
    
  


