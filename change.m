%% 定义方法
function [rgb]=change(img,m1,m2,m3)
%% r通道处理
r=double(img(:,:,m1));
%计算波段最大最小值
max_r=max(max(r));
min_r=min(min(r));
%放缩至0-255
r=uint8(255*(r-min_r)/(max_r-min_r));
%% g通道处理
g=double(img(:,:,m2));
max_g=max(max(g));
min_g=min(min(g));
g=uint8(255*(g-min_g)/(max_g-min_g));
%% b通道处理
b=double(img(:,:,m3));
max_b=max(max(b));
min_b=min(min(b));
b=uint8(255*(b-min_b)/(max_b-min_b));
%% 彩色合成
rgb=uint8(zeros(size(r,1),size(r,2),3));
rgb(:,:,1)=r;
rgb(:,:,2)=g;
rgb(:,:,3)=b;
