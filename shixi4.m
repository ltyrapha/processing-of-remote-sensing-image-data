[img,p,t]=freadenvi('D:\遥感数字图像处理\实习12\实习2\GF2-WHU');
img=img';
img=reshape(img,p(3),p(1),p(2));
r=img(1,:,:);
r=permute(r,[3,2,1]);
max_r=max(max(r));
min_r=min(min(r));
r = uint8(255*(r-min_r)/(max_r-min_r));
g=img(2,:,:);
g=permute(g,[3,2,1]);
max_g=max(max(g));
min_g=min(min(g));
g = uint8(255*(g-min_g)/(max_g-min_g));
b=img(3,:,:);
b=permute(b,[3,2,1]);
max_b=max(max(b));
min_b=min(min(b));
b = uint8(255*(b-min_b)/(max_b-min_b));
% 图像彩色合成
rgb=uint8(zeros(size(r,1),size(r,2),3));
rgb(:,:,1) = r;
rgb(:,:,2) = g;
rgb(:,:,3) = b;
% 彩色图像转化为灰度图像
gray = rgb2gray(rgb);
% 图像显示
figure(1),subplot(121),imshow(rgb), title('原始图像');
subplot(122),imshow(gray), title('灰度图像');
%% 阈值分割
T1=30;
Tb1=gray>T1;
T2=50;
Tb2=gray>T2;
figure(2);
subplot(2,2,1),imshow(gray),title('原始图像');
subplot(2,2,2),imhist(gray),title('直方图');
subplot(2,2,3),imshow(Tb1),title(['阈值分割:',num2str(T1)]);
subplot(2,2,4),imshow(Tb2),title(['阈值分割:',num2str(T2)]);
%% 阈值分割新方法
f=gray;
f=im2double(f);
%全局阈值
figure(3);
T=0.5*(min(f(:))+max(f(:)));
done=false;
while ~done
g=f>=T;
Tn=0.5*(mean(f(g))+mean(f(~g)));
done = abs(T-Tn)<0.1;
T=Tn;
end
r=imbinarize(f,T);
subplot(2,2,1);imshow(f),title('(a)原始图像');
subplot(2,2,2);imshow(r),title('(b)迭代法全局阈值分割');
Th=graythresh(f);%阈值
s=imbinarize(f,Th);
subplot(2,2,3);imshow(s),title('(c)全局阈值Otsu法阈值分割');
se=strel('disk',10);
ft=imtophat(f,se);
Thr=graythresh(ft);
lt = imbinarize(ft,Thr);
subplot(2,2,4);imshow(lt),title('(d)局部阈值分割');
%% 边缘分割
J1 = edge(gray,'roberts');
J2 = edge(gray,'sobel');
J3 = edge(gray,'Prewitt');
J4 = edge(gray,'log');
J5 = edge(gray,'canny');
% 图像显示
figure(4);
subplot(2,3,1),imshow(gray),title('原始图像');
subplot(2,3,2),imshow(J1),title('Roberts');
subplot(2,3,3),imshow(J2),title('Sobel');
subplot(2,3,4),imshow(J3),title('Prewitt');
subplot(2,3,5),imshow(J4),title('Log');
subplot(2,3,6),imshow(J5),title('Canny');

%% 数学形态学
I1 = 56;
Binary = gray > I1;
B = [1,1,1,1,1,1,1,1,1]; %模板B
Binary_dilate = imdilate(Binary, B); %膨胀
Binary_erode = imerode(Binary, B); %腐蚀
Binary_open = imopen(Binary, B); %开运算
Binary_close = imclose(Binary, B); %闭运算
% 图像显示
figure(5);
subplot(3,2,1),imshow(gray),title('原始图像');
subplot(3,2,2),imshow(Binary),title('二值图像');
subplot(3,2,3),imshow(Binary_dilate),title('膨胀');
subplot(3,2,4),imshow(Binary_erode),title('腐蚀');
subplot(3,2,5),imshow(Binary_open),title('开运算');
subplot(3,2,6),imshow(Binary_close),title('闭运算');
%% 二级腐蚀与膨胀
bd2 = imdilate(Binary_dilate,B);
be2 = imerode(Binary_erode,B);
figure(6);
subplot(2,3,1),imshow(gray),title('原始图像');
subplot(2,3,4),imshow(Binary),title('二值图像');
subplot(2,3,2),imshow(Binary_dilate),title('膨胀');
subplot(2,3,3),imshow(Binary_erode),title('腐蚀');
subplot(2,3,5),imshow(bd2),title('二级膨胀');
subplot(2,3,6),imshow(be2),title('二级腐蚀');
%% 二值图像骨架线提取以及粗化图像填充
BW = imbinarize(gray);
BW1 = bwmorph(BW,'remove');%remove形态学骨架提取
BW2 = bwmorph(BW,'skel',Inf);%skel形态学骨架提取
BW3 = bwperim(BW);%边界线提取
BW4 = imfill(BW,'hole');%填充
figure(7);
subplot(2,3,1),imshow(BW),title('原始二值化图像');
subplot(2,3,2),imshow(BW1),title('remove形态学骨架提取');
subplot(2,3,3),imshow(BW2),title('skel形态学骨架提取');
subplot(2,3,4),imshow(BW),title('原始二值化图像');
subplot(2,3,5),imshow(BW3),title('边界');
subplot(2,3,6),imshow(BW4),title('填充');
%% 附加
%加载VLFeat
run('D:\遥感数字图像处理\实习34\实习4\vlfeat-0.9.12\toolbox\vl_setup.m');
%Mean Shift分割
ratio=0.5;
kernelsize=8;
maxdist=10;
Iseg=vl_quickseg(rgb,ratio,kernelsize,maxdist);
%图像显示
figure(8);
subplot(1,2,1),imshow(rgb),title('彩色图像');
subplot(1,2,2),imshow(Iseg),title('分割图像');