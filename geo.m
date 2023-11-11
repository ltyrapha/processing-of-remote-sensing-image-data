%% r通道处理
[r,georef]=readgeoraster('D:\遥感数字图像处理\数据\wuhan\LC81230392016205LGN00_B4.TIF');%读取B4作为r通道数据
r=im2double(r);%将数据转换为double类型
r=mat2gray(r);%将r归一化到[0,1]区间内，最大最小值分别赋值1和0
r=im2uint8(r);%将r扩充到[0,255]
%% g通道处理
[g,georef]=readgeoraster('D:\遥感数字图像处理\数据\wuhan\LC81230392016205LGN00_B3.TIF');
g=im2double(g);
g=mat2gray(g);
g=im2uint8(g);
%% b通道处理
[b,georef]=readgeoraster('D:\遥感数字图像处理\数据\wuhan\LC81230392016205LGN00_B2.TIF');
b=im2double(b);
b=mat2gray(b);
b=im2uint8(b);
%% 彩色合成
rgb=uint8(zeros(size(r,1),size(r,2),3));
rgb(:,:,1)=r;
rgb(:,:,2)=g;
rgb(:,:,3)=b;
%% 图像显示
figure(1),imshow(rgb),title('wuhan');
imwrite(rgb,'D:\遥感数字图像处理\practice1\武汉_write_rgb.jpg','quality',95);