function clusterResult( k,p,path)
%UNTITLED14 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%% 
file = [];
num = size(p,1);
p1 = (p(:,1) == k);
for i=num:-1:1
    if(p1(i)==1)
        file = [file;i];
    end
end
%% 
%save file1.txt file1 -ascii 
%%

path_prex = [path,num2str(k)];
dlmwrite([path_prex,'.txt'],file);

end

