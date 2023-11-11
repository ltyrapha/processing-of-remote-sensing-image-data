%%图像读取、显示、写入
%利用imread读取图像
img=imread("D:\遥感数字图像处理\实习12\实习1\GF2_tiff.tif");
[rgb]=change(img,3,2,1);
%图像显示
figure(1);
imshow(rgb);
%图像写入
imwrite(rgb,'D:\遥感数字图像处理\practice1\GF2_Write_1.jpg','quality',95);

%利用geotiffread读取
[img,georef]=readgeoraster("D:\遥感数字图像处理\实习12\实习1\GF2_tiff.tif");
%图像显示
figure(2);
imshow(rgb);
%图像写入
imwrite(rgb,'D:\遥感数字图像处理\practice1\GF2_Write_2.jpg','quality',95);
%% 直方图统计
%读取文件
[img,georef]=readgeoraster('D:\遥感数字图像处理\实习12\实习1\GF2_tiff.tif');
%获取波段信息
b1=double(img(:,:,1));
b2=double(img(:,:,2));
b3=double(img(:,:,3));
b4=double(img(:,:,4));
%画图
figure(3);
subplot(2,2,1),h1=histogram(b1,50);title('原始直方图');
subplot(2,2,2),h2=histogram(b2,50);title('原始直方图');
subplot(2,2,3),h3=histogram(b3,50);title('原始直方图');
subplot(2,2,4),h4=histogram(b4,50);title('原始直方图');
%% 二值化
%设置二值化的分界值，小于300的都为0，大于300的都为1。
bw1=imbinarize(b1,300);
bw2=imbinarize(b2,300);
bw3=imbinarize(b3,300);
bw4=imbinarize(b4,300);
figure(4);
subplot(2,2,1),imshow(bw1),title('二值化');
subplot(2,2,2),imshow(bw2),title('二值化');
subplot(2,2,3),imshow(bw3),title('二值化');
subplot(2,2,4),imshow(bw4),title('二值化');
figure(5);
subplot(2,4,1),h1=histogram(b1,50);title('二值化前');
subplot(2,4,2),h1_h=histogram(bw1);title('二值化后');
subplot(2,4,3),h2=histogram(b2,50);title('二值化前');
subplot(2,4,4),h2_h=histogram(bw2);title('二值化后');
subplot(2,4,5),h3=histogram(b3,50);title('二值化前');
subplot(2,4,6),h3_h=histogram(bw3);title('二值化后');
subplot(2,4,7),h4=histogram(b4,50);title('二值化前');
subplot(2,4,8),h4_h=histogram(bw4);title('二值化后');
%% 最大最小拉伸
max_b1=max(max(b1));
min_b1=min(min(b1));
b1_maxmin=uint8(255*(b1-min_b1)/(max_b1-min_b1));
max_b2=max(max(b2));
min_b2=min(min(b2));
b2_maxmin=uint8(255*(b2-min_b2)/(max_b2-min_b2));
max_b3=max(max(b3));
min_b3=min(min(b3));
b3_maxmin=uint8(255*(b3-min_b3)/(max_b3-min_b3));
max_b4=max(max(b4));
min_b4=min(min(b4));
b4_maxmin=uint8(255*(b4-min_b4)/(max_b4-min_b4));
figure(6);
subplot(2,2,1),imshow(b1_maxmin),title('最大最小拉伸');
subplot(2,2,2),imshow(b2_maxmin),title('最大最小拉伸');
subplot(2,2,3),imshow(b3_maxmin),title('最大最小拉伸');
subplot(2,2,4),imshow(b4_maxmin),title('最大最小拉伸');
%% 2%拉伸
Bincount=h1.BinCounts;
Binedge=h1.BinEdges;
Percbin=Bincount/sum(Bincount);
Cumpercbin1=cumsum(Percbin);
Cumpercbin2=cumsum(flip(Percbin));
index=find(Cumpercbin1<0.02);
if index
min_stren=Binedge(index(end)+1) ;
else
min_stren=0;
end
index=find(Cumpercbin2<0.02);
if index
max_stren=Binedge(end-index(end));
else
max_stren=0;
end
b1_stren=uint8(255*(b1-min_stren)/(max_stren-min_stren));

Bincount=h2.BinCounts;
Binedge=h2.BinEdges;
Percbin=Bincount/sum(Bincount);
Cumpercbin1=cumsum(Percbin);
Cumpercbin2=cumsum(flip(Percbin));
index=find(Cumpercbin1<0.02);
if index
min_stren=Binedge(index(end)+1) ;
else
min_stren=0;
end
index=find(Cumpercbin2<0.02);
if index
max_stren=Binedge(end-index(end));
else
max_stren=0;
end
b2_stren=uint8(255*(b2-min_stren)/(max_stren-min_stren));

Bincount=h3.BinCounts;
Binedge=h3.BinEdges;
Percbin=Bincount/sum(Bincount);
Cumpercbin1=cumsum(Percbin);
Cumpercbin2=cumsum(flip(Percbin));
index=find(Cumpercbin1<0.02);
if index
min_stren=Binedge(index(end)+1) ;
else
min_stren=0;
end
index=find(Cumpercbin2<0.02);
if index
max_stren=Binedge(end-index(end));
else
max_stren=0;
end
b3_stren=uint8(255*(b3-min_stren)/(max_stren-min_stren));

Bincount=h4.BinCounts;
Binedge=h4.BinEdges;
Percbin=Bincount/sum(Bincount);%计算概率
Cumpercbin1=cumsum(Percbin);%正序
Cumpercbin2=cumsum(flip(Percbin));%逆序
index=find(Cumpercbin1<0.02);%找到前2%
if index
min_stren=Binedge(index(end)+1) ;
else
min_stren=0;
end
index=find(Cumpercbin2<0.02);%找到后2%
if index
max_stren=Binedge(end-index(end));
else
max_stren=0;
end
b4_stren=uint8(255*(b4-min_stren)/(max_stren-min_stren));

figure(8),subplot(2,2,1),imshow(b1_stren),title('2%拉伸');
figure(8),subplot(2,2,2),imshow(b2_stren),title('2%拉伸');
figure(8),subplot(2,2,3),imshow(b3_stren),title('2%拉伸');
figure(8),subplot(2,2,4),imshow(b4_stren),title('2%拉伸');
%% 直方图均衡化
%对各波段进行均衡化
b1_eq = histeq(b1_maxmin);
b2_eq = histeq(b2_maxmin);
b3_eq = histeq(b3_maxmin);
b4_eq = histeq(b4_maxmin);
%均衡化后图像显示
figure(9);
subplot(2,2,1),imshow(b1_eq),title('直方图均衡化');
subplot(2,2,2),imshow(b2_eq),title('直方图均衡化');
subplot(2,2,3),imshow(b3_eq),title('直方图均衡化');
subplot(2,2,4),imshow(b4_eq),title('直方图均衡化');
%均衡化后直方图显示
figure(10);
subplot(2,2,1),hh1 = histogram(b1_eq);title('直方图均衡化');
subplot(2,2,2),hh2 = histogram(b2_eq);title('直方图均衡化');
subplot(2,2,3),hh3 = histogram(b3_eq);title('直方图均衡化');
subplot(2,2,4),hh4 = histogram(b4_eq);title('直方图均衡化');
%% 直方图匹配
%对band4进行直方图匹配
b4_histmatch = imhistmatch(b4_maxmin,b2_maxmin);
%画出原始图像、目标图像和匹配后图像
figure(11);
subplot(1,3,1),imshow(b4_maxmin),title('原始图像');
subplot(1,3,2),imshow(b2_maxmin),title('目标图像');
subplot(1,3,3),imshow(b4_histmatch),title('直方图匹配后图像');
%画出原始直方图、目标直方图和匹配后直方图
figure(12);
subplot(1,3,1),h4 = histogram(b4,50);title('原始直方图')
subplot(1,3,2),h2 = histogram(b2,50);title('目标直方图');
subplot(1,3,3),h4_2 = histogram(b4_histmatch,50);title('匹配后直方图');

