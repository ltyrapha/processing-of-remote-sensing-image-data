%% 遥感影像的读取、显示与保存
[img,p,t]=freadenvi('D:\遥感数字图像处理\实习12\实习2\GF2-WHU');
img=img';
img=reshape(img,p(3),p(1),p(2));
%红绿蓝波段分别读取，并进行最大最小值拉伸
r=img(1,:,:);
r=permute(r,[3,2,1]);
max_r=max(max(r));
min_r=min(min(r));
r=uint8(255*(r-min_r)/(max_r-min_r));
g=img(2,:,:);
g=permute(g,[3,2,1]);
max_g=max(max(g));
min_g=min(min(g));
g=uint8(255*(g-min_g)/(max_g-min_g));
b=img(3,:,:);
b=permute(b,[3,2,1]);
max_b=max(max(b));
min_b=min(min(b));
b=uint8(255*(b-min_b)/(max_b-min_b));
%彩色合成
rgb=uint8(zeros(size(r,1),size(r,2),3));
rgb(:,:,1)=r;
rgb(:,:,2)=g;
rgb(:,:,3)=b;
%图像显示
figure(1);
imshow(rgb);
%图像写入
% imwrite(rgb,'D:\遥感数字图像处理\practice2\GF2_write1.jpg','quality',95);
%% 图像空间滤波
%将彩色图像转换为灰度图像
imgGray=rgb2gray(rgb);
%添加椒盐噪声
saltImg=imnoise(imgGray,"salt & pepper");
%添加高斯噪声
gussImg=imnoise(imgGray,"gaussian");
%图像显示
figure(2);
subplot(1,3,1),imshow(imgGray),title('原始图像');
subplot(1,3,2),imshow(saltImg),title('椒盐噪声');
subplot(1,3,3),imshow(gussImg),title('高斯噪声');

% %图像写入
% imwrite(imgGray,'D:\遥感数字图像处理\practice2\imgGray-Write.jpg','quality',95);
% imwrite(saltImg,'D:\遥感数字图像处理\practice2\saltImg-Write.jpg','quality',95);
% imwrite(GussImg,'D:\遥感数字图像处理\practice2\GussImg-Write.jpg','quality',95);

%生成滤波模板
h1=[1/9, 1/9, 1/9; 1/9, 1/9, 1/9; 1/9, 1/9, 1/9;];%手动生成3*3的均值滤波模板
h2=fspecial('average',3);%函数生成3*3的均值滤波模板
h3=fspecial('average',5);%函数生成5*5的均值滤波模板
%均值滤波处理椒盐噪声
avrg3=imfilter(saltImg,h2,'corr','replicate');
avrg5=imfilter(saltImg,h3,'corr','replicate');
%中值滤波处理椒盐噪声
midr3=medfilt2(saltImg,[3 3]);
midr5=medfilt2(saltImg,[5 5]);

%图像显示
figure(3);
subplot(2,3,1),imshow(imgGray),title('原始图像');
subplot(2,3,2),imshow(saltImg),title('椒盐噪声');
subplot(2,3,3),imshow(avrg3),title('3*3 均值');
subplot(2,3,4),imshow(avrg5),title('5*5 均值');
subplot(2,3,5),imshow(midr3),title('3*3 中值');
subplot(2,3,6),imshow(midr5),title('5*5 中值');

%均值滤波处理高斯噪声
avrg3=imfilter(gussImg,h2,'corr','replicate');
avrg5=imfilter(gussImg,h3,'corr','replicate');
%中值滤波处理高斯噪声
midr3=medfilt2(gussImg,[3 3]);
midr5=medfilt2(gussImg,[5 5]);

%图像显示
figure(4);
subplot(2,3,1),imshow(imgGray),title('原始图像');
subplot(2,3,2),imshow(gussImg),title('高斯噪声');
subplot(2,3,3),imshow(avrg3),title('3*3 均值');
subplot(2,3,4),imshow(avrg5),title('5*5 均值');
subplot(2,3,5),imshow(midr3),title('3*3 中值');
subplot(2,3,6),imshow(midr5),title('5*5 中值');

%生成Laplacian模板
h4=[0, -1, 0; -1, 4, -1; 0, -1, 0;];%手动生成3*3的锐化算子
h5=fspecial('unsharp',0);%函数生成3*3的锐化滤波模板
h6=fspecial('laplacian',0);%生成Laplacian边缘模板
%图像处理
lapls_sharp=imfilter(imgGray,h5,'corr','replicate');%拉普拉斯锐化
lapls_border=imfilter(imgGray,h6,'corr','replicate');%拉普拉斯边缘检测
%显示图像
figure(5);
subplot(1,3,1),imshow(imgGray),title('原始图像');
subplot(1,3,2),imshow(lapls_sharp),title('Laplacian锐化');
subplot(1,3,3),imshow(lapls_border),title('Laplacian边缘检测');
% 
% %图像写入
% imwrite(lapls_sharp,'D:\遥感数字图像处理\practice2\lapls_sharp-Write.jpg','quality',95);
% imwrite(lapls_border,'D:\遥感数字图像处理\practice2\lapls_border-Write.jpg','quality',95);

%% 二维快速傅里叶变换
%彩色图像变换成灰度图像
imgGray=rgb2gray(rgb);
%数据格式转换
I_low=double(imgGray);
%二维傅立叶变换
F=fft2(I_low);
%频率域谱的中心化
F=fftshift(F);
%频率域谱的拉伸
F=abs(F); 
T=log(F+1);
% %图像显示
figure(6);
subplot(3,2,1),imshow(uint8(imgGray)),title('原始图像');
subplot(3,2,2),imshow(T,[]),title('原始图像频谱图');
%% 加入噪声的图像频率谱
%在图像中加入密度为0.04的椒盐噪声 
S=imnoise(imgGray,'salt & pepper',0.04); 
K=fft2(double(S)); 
K=fftshift(K); 
K=abs(K); 
T=log(K+1);
subplot(3,2,3),imshow(uint8(S)),title('椒盐噪声');
subplot(3,2,4),imshow(T,[]),title('椒盐噪声频谱图');
%在图像中加入均值为0，方差为2的高斯噪声
G=imnoise(imgGray,'gaussian',0,0.02);
H=fft2(double(G));
H=fftshift(H); H=abs(H); 
T=log(H+1);
subplot(3,2,5),imshow(uint8(G)),title('高斯噪声');
subplot(3,2,6),imshow(T,[]),title('高斯噪声频谱图');

%% 理想高通滤波与低通滤波
[M,N]=size(S);
S_hp=S;
S_lp=S;
square=sqrt(M^2+N^2);
D0=square/30;%截止频率
for i=1:M
    for j=1:N
        if sqrt((i-M/2)^2+(j-N/2)^2)>D0
            h=1;
        else
            h=0;
        end
        S_hp(i,j)=S(i,j)*h; %高通滤波
        S_lp(i,j)=S(i,j)*(1-h); %低通滤波
    end
end
%对滤波之后的图像，进行傅立叶逆变换
S_high1=abs(S_hp);
S_high1=log(S_high1+1);

S_low1=abs(S_lp);
S_low1=log(S_low1+1);

S_hp = ifftshift(S_hp);
S_lp = ifft2(S_lp);
I_high = uint8(real(S_hp));

S_lp = ifftshift(S_lp);
S_lp = ifft2(S_lp);
I_low = uint8(real(S_lp));

figure(7);
subplot(3,2,1),imshow(imgGray),title('原始图像');
S=fftshift(fft2(double(imgGray)));
Simg=abs(S);
Simg=log(Simg+1);
subplot(3,2,2),imshow(Simg,[]),title('原始图像频谱图');
subplot(3,2,3),imshow(I_high,[]),title('理想高通滤波图像');
subplot(3,2,4),imshow(S_high1,[]),title('理想高通滤波频谱图');
subplot(3,2,5),imshow(I_low,[]),title('理想低通滤波图像');
subplot(3,2,6),imshow(S_low1,[]),title('理想低通滤波频谱图');
%% 巴特沃高通滤波及低通滤波
D0 = square/30;
S_lpbtw = S;
S_hpbtw = S;
for i = 1:M
    for j = 1:N
n = 1; 
S_lpbtw(i,j) = (1/(1+((sqrt((i-M/2)^2+(j-N/2)^2))/D0)^(2*n)))*S(i,j);
S_hpbtw(i,j) = (1/(1+(D0/(sqrt((i-M/2)^2+(j-N/2)^2)))^(2*n)))*S(i,j);
    end
end
S_hpbtw1 = abs(S_hpbtw);
S_hpbtw1 = log(S_hpbtw1 + 1);
S_hpbtw =ifftshift(S_hpbtw);
I_hpbtw = uint8(real(ifft2(S_hpbtw)));

S_lpbtw1 = abs(S_lpbtw);
S_lpbtw1 = log(S_lpbtw1 + 1);
S_lpbtw =ifftshift(S_lpbtw);
I_lpbtw = uint8(real(ifft2(S_lpbtw)));

figure(8);
subplot(3,2,1),imshow(imgGray),title('原始图像');
subplot(3,2,2),imshow(Simg,[]),title('原始图像频谱图');
subplot(3,2,3),imshow(I_hpbtw),title('巴特沃高通滤波图像');
subplot(3,2,4),imshow(S_hpbtw1,[]),title('巴特沃高通滤波图像频谱图');
subplot(3,2,5),imshow(I_lpbtw),title('巴特沃低通滤波图像');
subplot(3,2,6),imshow(S_lpbtw1,[]),title('巴特沃低通滤波图像频谱图');
%% 指数高通滤波及低通滤波
D0 = square/30;
S_lpexp = S;
S_hpexp = S;
for i = 1:M
    for j = 1:N
n = 1; 
S_lpexp(i,j) = exp(-(sqrt((i-M/2)^2+(j-N/2)^2)/D0)^n)*S(i,j);
S_hpexp(i,j) = exp(-D0/(sqrt((i-M/2)^2+(j-N/2)^2))^n)*S(i,j);
    end
end
S_hpexp1 = abs(S_hpexp);
S_hpexp1 = log(S_hpexp1 + 1);
S_hpexp =ifftshift(S_hpexp);
I_hpexp = uint8(real(ifft2(S_hpexp)));

S_lpexp1 = abs(S_lpexp);
S_lpexp1 = log(S_lpexp1 + 1);
S_lpexp =ifftshift(S_lpexp);
I_lpexp = uint8(real(ifft2(S_lpexp)));

figure(9);
S=fftshift(fft2(double(imgGray)));
Simg=abs(S);
Simg=log(Simg+1);
subplot(3,2,1),imshow(imgGray),title('原始图像');
subplot(3,2,2),imshow(Simg,[]),title('原始图像频谱图');
subplot(3,2,3),imshow(I_hpexp),title('指数高通滤波图像');
subplot(3,2,4),imshow(S_hpexp1,[]),title('指数高通滤波图像频谱图');
subplot(3,2,5),imshow(I_lpexp),title('指数低通滤波图像');
subplot(3,2,6),imshow(S_lpexp1,[]),title('指数低通滤波图像频谱图');

figure(10);
subplot(2,4,1),imshow(imgGray),title('原始图像');
subplot(2,4,2),imshow(I_high,[]),title('理想高通滤波图像');
subplot(2,4,3),imshow(I_hpbtw),title('巴特沃高通滤波图像');
subplot(2,4,4),imshow(I_hpexp),title('指数高通滤波图像');
subplot(2,4,5),imshow(Simg,[]),title('原始图像频谱图');
subplot(2,4,6),imshow(S_high1,[]),title('理想高通滤波频谱图');
subplot(2,4,7),imshow(S_hpbtw1,[]),title('巴特沃高通滤波图像频谱图');
subplot(2,4,8),imshow(S_hpexp1,[]),title('指数高通滤波图像频谱图');
figure(11);
subplot(2,4,1),imshow(imgGray),title('原始图像');
subplot(2,4,2),imshow(I_low,[]),title('理想低通滤波图像');
subplot(2,4,3),imshow(I_lpbtw),title('巴特沃低通滤波图像');
subplot(2,4,4),imshow(I_lpexp),title('指数低通滤波图像');
subplot(2,4,5),imshow(Simg,[]),title('原始图像频谱图');
subplot(2,4,6),imshow(S_lpbtw1,[]),title('巴特沃低通滤波图像频谱图');
subplot(2,4,7),imshow(S_low1,[]),title('理想低通滤波频谱图');
subplot(2,4,8),imshow(S_lpexp1,[]),title('指数低通滤波图像频谱图');
