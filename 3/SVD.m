%% 
clc;
clear;
%% 
datatmp = load ('ratings.mat');
%ԭʼ���ݣ��û���|��Ӱ��|����|ʱ���
data = datatmp.ratings;

%%
% rating_matrix���û��Ե�Ӱ�����֡��У��û����У���Ӱ��6040*3952
row = size(data,1);
rating_matrix = zeros(6040,3952);
for index=1:row
    i = data(index,1);
    j = data(index,2);
    rating = data(index,3);
    rating_matrix(i,j) = rating;
end

%% 
trainingdata_tmp = load ('trainfile.mat');
trainingdata = trainingdata_tmp.trainfile;

training_rating = zeros(6040,3952);
for index=1:size(trainingdata,1)
    i = trainingdata(index,1);
    j = trainingdata(index,2);
    rating = data(index,3);
    training_rating(i,j) = rating;
end
[U,S,V]= svd(training_rating);
%% 
tempMatrix = S.^2;
sum_before = sum(tempMatrix(:));
for k = 1000:-1:1
    temp = tempMatrix(1:k,1:k);
    sum_after = sum(temp(:));
    if(sum_after/sum_before<0.9)
        break;
    end
end
disp(k);
%%
%dimension
dim=k;
%U_new��ʾ�û���������
U_new = U(:,1:dim);
S_new = S(1:dim,1:dim);
%V_new��ʾ��Ӱ��������
V_new = V(:,1:dim);

result = [num2str(dim),',',num2str(size(data,2))];
fid=fopen('3.txt','wt');
fprintf(fid,'%s\n',result);
fclose(fid);

save userTraining U_new;
save movieTraining V_new;

%movies 3952*969 ��ά  ��ά������ݼ� based on movies 
movies = rating_matrix'*U_new*S_new^-1;

save movieData movies;
%% 
%users 6040*969 ��ά ��ά������ݼ� based on users
users = rating_matrix*V_new*S_new^-1;
%% 
%plot(users(:,1),plot(users,2));
save userData users;
%% 

