%% 
clc;
clear;
%% 
datatmp = load ('ratings.mat');
%原始数据，用户号|电影号|评分|时间戳
data = datatmp.ratings;

%%
% rating_matrix是用户对电影的评分。行：用户；列：电影。6040*3952
row = size(data,1);
rating_matrix = zeros(6040,3952);
for index=1:row
    i = data(index,1);
    j = data(index,2);
    rating = data(index,3);
    rating_matrix(i,j) = rating;
end
%save rating_matrix;
%% kmeans
k=20;
initCentroid = rating_matrix(1:k,1:end);
opts = statset('Display','final');
[p,C] = kmeans(rating_matrix,k,'Distance','sqeuclidean','Start',initCentroid,'Options',opts);
%[p,C] = kMeans(rating_matrix,k);
%% 
theta = 0;
for i=1:6040
    cluster_num = p(i);
    centroid = C(cluster_num,:);
    temp = rating_matrix(i,:)-centroid;
    theta = theta + norm(temp,2)^2;
end
disp(['the cost is ', num2str(theta)]);
 
mkdir('result_20')
for i=1:k
    clusterResult(i,p,result_20/);
end

