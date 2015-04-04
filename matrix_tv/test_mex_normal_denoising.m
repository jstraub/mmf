% run in folder before running matlab 
% export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/build


I=imread('cameraman.tif');I=single(I)/255;
I=imresize(I,[384 256]);
sig_n=I;
sig_I=I;
I0=I;
resI=I;
I0(:,:,2)=1-I0(:,:,1);
I0(:,:,3)=I0(:,:,1);
I0=single(bsxfun(@rdivide,I0,sqrt(sum(I0.^2,3))));
mex_normal_denoising(I0,single(0.05));
close all;
imshow(I0,[]);
