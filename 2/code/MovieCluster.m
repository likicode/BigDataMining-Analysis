%% 
clc;
clear;
%% 
datatmp = load ('ratings.mat');
%原始数据，用户号|电影号|评分|时间戳
data = datatmp.ratings;

%%
% rating_matrix是用户对电影的评分。行：电影；列：用户。3952*6040
row = size(data,1);
rating_matrix = zeros(3952,6040);
for index=1:row
    i = data(index,2);
    j = data(index,1);
    rating = data(index,3);
    rating_matrix(i,j) = rating;
end

%% kmeans
k=20;
initCentroid = rating_matrix(1:k,1:end);
opts = statset('Display','final');
[p,C] = kmeans(rating_matrix,k,'Distance','sqeuclidean','Start',initCentroid,'Options',opts);
%[p,C] = kMeans(rating_matrix,k);
%% 
theta = 0;
for i=1:3952
    cluster_num = p(i);
    centroid = C(cluster_num,:);
    temp = rating_matrix(i,:)-centroid;
    theta = theta + norm(temp,2)^2;
end
disp(['the cost is ', num2str(theta)]);
%% 
mkdir('movies_c20')
for i=1:k
    clusterResult(i,p,'movies_c20/');
end
