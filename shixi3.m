clc;clear;
%图像读取
A = imread('D:\遥感数字图像处理\实习34\实习3\GF2-MSS.tif');
B = imread('D:\遥感数字图像处理\实习34\实习3\GF2-PAN.tif');
r = A(:,:,3);
g = A(:,:,2);
b = A(:,:,1);
h = A(:,:,4);
% %重采样
% x = imresize(r,[1600 1600],'bilinear');
% y = imresize(g,[1600 1600],'bilinear');
% z = imresize(b,[1600 1600],'bilinear');

% %加权平均
% c1 = (0.4*x+0.6*B);
% c2 = (0.4*y+0.6*B);
% c3 = (0.4*z+0.6*B);
%相乘法
% x=double(x);
% y=double(y);
% z=double(z);
% B=double(B);
% c1=x.*B;
% c2=y.*B;
% c3=z.*B;
% %Brovey法
% x=double(x);
% y=double(y);
% z=double(z);
% B=double(B);
% c1=x.*B./(x+y+z);
% c2=y.*B./(x+y+z);
% c3=z.*B./(x+y+z);

% re_mss = imresize(A,2,'bicubic');
% figure(1);
% 
% r=double(re_mss(:,:,3));
% max_r=max(max(r));
% min_r=min(min(r));
% r=uint8(255*(r-min_r)/(max_r-min_r));
% 
% g=double(re_mss(:,:,2));
% max_g=max(max(g));
% min_g=min(min(g));
% g=uint8(255*(g-min_g)/(max_g-min_g));
% 
% b=double(re_mss(:,:,1));
% max_b=max(max(b));
% min_b=min(min(b));
% b=uint8(255*(b-min_b)/(max_b-min_b));
% 
% rgb=uint8(zeros(size(r,1),size(r,2),3));
% rgb(:,:,1)=r;
% rgb(:,:,2)=g;
% rgb(:,:,3)=b;
% 
% subplot(1,2,1);imshow(rgb);title('重采样');
% 
% r_ronghe = uint8(255 * (c1-min(min(c1)))./(max(max(c1))-min(min(c1))));
% g_ronghe = uint8(255 * (c2-min(min(c2)))./(max(max(c2))-min(min(c2))));
% b_ronghe = uint8(255 * (c3-min(min(c3)))./(max(max(c3))-min(min(c3))));
% rgb_ronghe = uint8(zeros(size(r_ronghe,1),size(g_ronghe,2),3));
% rgb_ronghe(:,:,1) = r_ronghe;
% rgb_ronghe(:,:,2) = g_ronghe;
% rgb_ronghe(:,:,3) = b_ronghe;
% 
% subplot(1,2,2);imshow(rgb_ronghe);title('图像融合');

% %IHS方法
% %正变换
% A1=imresize(A,4);
% B=double(B);
% B=uint8(255*(B-min(min(B)))./(max(max(B))-min(min(B))));%拉伸需要原始数据为非整型
% A1=double(A1);
% A1=uint8(255*(A1-min(min(A1)))./(max(max(A1))-min(min(A1))));
% A1rgb=uint8(zeros(1600,1600,3));
% A1rgb(:,:,3)=A1(:,:,1);
% A1rgb(:,:,2)=A1(:,:,2);
% A1rgb(:,:,1)=A1(:,:,3);
% r=A1rgb(:,:,1);
% g=A1rgb(:,:,2);
% b=A1rgb(:,:,3);
% r=double(r);
% g=double(g);
% b=double(b);
% ang=acosd((1/2)*(2*r-g-b)./sqrt((r-g).^2+(r-b).*(g-b)));
% H=double(zeros(1600,1600,1));       
% for i=1:1600
%      for j=1:1600
%         if(g(i,j)>=b(i,j))
%              H(i,j)=ang(i,j);
%          else
%              H(i,j)=360-ang(i,j);
%          end
%      end
% end
% I=(r+g+b)/3;
% a=min(A1rgb,[],3);
% a=double(a);
% S=1-3*a./(r+b+g);
% %全色影像和亮度分量直方图匹配
% B=uint8(B);
% I=uint8(I);
% IB_ronghe=double(imhistmatch(B,I));
% %逆变换
% for i=1:1600
%     for j=1:1600
%     if(H(i,j)>=0&&H(i,j)<120)
%     b(i,j)=IB_ronghe(i,j)*(1-S(i,j));
%     g(i,j)=3*IB_ronghe(i,j)-r(i,j)-b(i,j);    
%     r(i,j)=IB_ronghe(i,j)*(1+S(i,j)*cosd(H(i,j))/cosd(60-H(i,j)));
%     end
%     if(H(i,j)>=120&&H(i,j)<240)
%     r(i,j)=IB_ronghe(i,j)*(1-S(i,j));
%     b(i,j)=3*IB_ronghe(i,j)-r(i,j)-g(i,j);
%     g(i,j)=IB_ronghe(i,j)*(1+S(i,j)*cosd(H(i,j)-120)/cosd(180-H(i,j)));
%     end
%     if(H(i,j)>=240&&H(i,j)<360)
%     r(i,j)=3*IB_ronghe(i,j)-g(i,j)-b(i,j);
%     g(i,j)=IB_ronghe(i,j)*(1-S(i,j));
%     b(i,j)=IB_ronghe(i,j)*(1+S(i,j)*cosd(H(i,j)-240)/cosd(300-H(i,j)));
%     end   
%     end
% end 
% rgbihs=double(zeros(1600,1600,3));
% rgbihs(:,:,1)=r;
% rgbihs(:,:,2)=g;
% rgbihs(:,:,3)=b;
% rgbihs=uint8(255*(rgbihs-min(min(rgbihs)))./(max(max(rgbihs))-min(min(rgbihs))));
% imshow(rgbihs);

%PCA融合方法
%求协方差矩阵
A=imresize(A,4);
A=double(A);
r=A(:,:,1);
g=A(:,:,2);
b=A(:,:,3);
h=A(:,:,4);
%将矩阵变成列向量，求协方差矩阵
x=cov([r(:) g(:) b(:) h(:)]);
%计算特征值和向量
[V,D]=eig(x);
%计算主成分
Y1=double(zeros(1600,1600,1));
Y2=double(zeros(1600,1600,1));
Y3=double(zeros(1600,1600,1));
Y4=double(zeros(1600,1600,1));
for i=1:1600
    for j=1:1600
    Y1(i,j)=V(:,4)'*[A(i,j,1);A(i,j,2);A(i,j,3);A(i,j,4)];
    Y2(i,j)=V(:,3)'*[A(i,j,1);A(i,j,2);A(i,j,3);A(i,j,4)];
    Y3(i,j)=V(:,2)'*[A(i,j,1);A(i,j,2);A(i,j,3);A(i,j,4)];
    Y4(i,j)=V(:,1)'*[A(i,j,1);A(i,j,2);A(i,j,3);A(i,j,4)];
    end
end
%匹配
B=double(B);
B1=(B-mean(mean(B)))*std2(Y1)/std2(B)+mean(mean(Y1));
%逆变换
for i=1:1600
    for j=1:1600
        A(i,j,1)=V(1,:)*[Y4(i,j);Y3(i,j);Y2(i,j);B1(i,j)];
        A(i,j,2)=V(2,:)*[Y4(i,j);Y3(i,j);Y2(i,j);B1(i,j)];
        A(i,j,3)=V(3,:)*[Y4(i,j);Y3(i,j);Y2(i,j);B1(i,j)];
        A(i,j,4)=V(4,:)*[Y4(i,j);Y3(i,j);Y2(i,j);B1(i,j)];
    end
end
A=uint8(255*(A-min(min(A)))./(max(max(A))-min(min(A))));
rgbpca=uint8(zeros(1600,1600,3));
rgbpca(:,:,1)=A(:,:,3);
rgbpca(:,:,2)=A(:,:,2);
rgbpca(:,:,3)=A(:,:,1);
imshow(rgbpca);

%精度评价，以pca为例，rmse和cc指标
rgb_ronghe=imresize(rgb_ronghe,0.25);
MSS=imread('GF2-MSS.tif');
MSS=double(MSS);
MSS=uint8(255*(MSS-min(min(MSS)))./(max(max(MSS))-min(min(MSS))));
MSSrgb=uint8(zeros(400,400,3));
MSSrgb(:,:,3)=MSS(:,:,1);
MSSrgb(:,:,2)=MSS(:,:,2);
MSSrgb(:,:,1)=MSS(:,:,3);
rgb_ronghe=double(rgb_ronghe);
MSSrgb=double(MSSrgb);
%RMSE
t=double(zeros(400,400,3));
t=(rgb_ronghe-MSSrgb).^2;
rmse=sqrt(sum(sum(t))/(400*400));
%CC
cc1=corrcoef(rgb_ronghe(:,:,1),MSSrgb(:,:,1));
cc2=corrcoef(rgb_ronghe(:,:,2),MSSrgb(:,:,2));
cc3=corrcoef(rgb_ronghe(:,:,3),MSSrgb(:,:,3));




