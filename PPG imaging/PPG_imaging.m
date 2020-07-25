% Photoplethysmographic imaging of high spatial resolution

% Steps to run the code:
% Select the ROI 
% Run the code - at breakpoints either you have to enter some values 
% or a plot will be shown
% Enter heart pulse wave bandwidth from the FT plot that appears when breakpoint occurs
% Blood perfusion during the cardiac cycle is visualized 

% Steps remaining:
% Adaptive algorithm

clear all;
close all;
clc;
pixel=1280;
skip=0;
line=1024;
ED_num=200;

filename=sprintf('sampleData.txt');
fid=fopen(filename,'r');

%% Testing for ROI

% Uncomment below code to test- enter subarea pixel values in line 2 below
% test = fread(fid,[pixel line],'uint8');
% test = test(:,:);     % Enter pixel values here 
% shft = 0*pixel*line;
% status = fseek(fid, shft, 'cof');   
% testsample=im2double(test);
% testnorm=(max(testsample(:))- min(testsample(:)))*testsample;
% test_aft_norm = mat2gray(testnorm);
% % pixel_values = impixel(test_aft_norm);
% imshow(test_aft_norm);
% axis on;

%%  -----------------------Select ROI Manually----------

% Select pixel values to cover the ROI
%             left_y           right_Y
%   up_x          ---------------- 
%                |                |
%                |      ROI       |
%   down_x        ---------------- 

up_x = 250;
down_x = 350;
left_y = 450;
right_y = 550;


%%  Finding the mean pixel value time trace and its fourier transform

% Processing input images
height = down_x-up_x +1;
width = right_y -left_y +1;
raw_data = zeros( height,width ,ED_num);
for ii=1:ED_num
	A = fread(fid,[pixel line],'uint8');
    A = A(up_x:down_x, left_y:right_y);
	sample=im2double(A(:,:));
	norm =(max(sample(:))- min(sample(:)))*sample;
	raw_data(:,:,ii)=mat2gray(norm);
	ii
end

mean_roi = zeros(ED_num);
for i = 1 : ED_num
    mean_roi(i) = mean(mean(raw_data(:,:,i)));
end

% Detrending DC level in the mean-pixel-value
dc_level = mean(mean_roi);
mean_roi = mean_roi - dc_level;

% plot(1:ED_num, mean_roi)

% Finding FT
Y = fft(mean_roi);
L = 200;
Fs = 148;
P2 = abs(Y/L);
P1 = P2(1:L/2+1); % Since the FT is symmetric about its half-point
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,P1) 
title('Power Spectrum of mean pixel value time-trace')
xlabel('f (Hz)')
ylabel('Magnitude')


%% ------------Select the frequency bands C1 and C2---------

C1 = 0.7;
C2 = 1.5;


%% Inverse FT of truncated band giving reference function Rf(t) 
% and normalizing it
Rf = ifft(Y(floor(C1*Fs): floor(C2*Fs)),ED_num);

subplot(2,1,1)
plot(mean_roi)
title("Mean pixel value time trace")

subplot(2,1,2)
hold on
plot(real(Rf),"r")
plot(imag(Rf),"b--")
title("Real(red) and Imaginary(blue) parts of reference function")
hold off

%% Calculation of correlation matrix S
S = zeros(height,width);
for i = 1 : ED_num
    S = S + raw_data(:,:,i)* Rf(i);
end    

imshow(abs(S))
title("Correlation matrix S");

    
%% Calculating Hc(x,y,t) matrix and showing dynamic changes of the blood 
% volume pulsations
phi = zeros(ED_num);
fc = (C1+C2)/2;
for i = 1 : ED_num
    phi(i) = 2 * 3.14 * fc * i;
end 

H = zeros(height,width,ED_num);
for i = 1 : ED_num
    H(:,:,i) = real(S) * cos(phi(i)) + imag(S) * sin(phi(i));
end 
H = H+0.5; % For easy shifting of negative values
% Final visualization of the blood perfusion during the cardiac cycle
implay(H)
