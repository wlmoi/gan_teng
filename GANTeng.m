%% ============================================================
%  FACE-GAN (CelebA-style Dataset)
%  ============================================================
%  Folders:
%   - dataset/training/*.jpg
%   - dataset/validation/*.jpg
%   - dataset/testing/*.jpg
%
%  Features:
%   - Generator & Discriminator with bias, ReLU, tanh
%   - 40 Epoch Training
%   - Validation & Testing evaluation
%   - Hardware simulation (fixed-point Q7.8)
%   - MSE, error %, and clock estimation
%  ============================================================

clear; clc; close all;

%% ================================
% 1. Load Datasets
% ================================
baseDir = 'dataset';
trainDir = fullfile(baseDir,'training');
valDir   = fullfile(baseDir,'validation');
testDir  = fullfile(baseDir,'testing');

trainFiles = dir(fullfile(trainDir,'*.jpg'));
valFiles   = dir(fullfile(valDir,'*.jpg'));
testFiles  = dir(fullfile(testDir,'*.jpg'));

numTrain = numel(trainFiles);
numVal   = numel(valFiles);
numTest  = numel(testFiles);

fprintf('Loaded: %d training, %d validation, %d testing images.\n', ...
        numTrain, numVal, numTest);

imgSize = [64 64];
for i = 1:numTrain
    img = im2double(imresize(imread(fullfile(trainDir,trainFiles(i).name)), imgSize));
    X_train(:,i) = reshape(img, [], 1);
end
for i = 1:numVal
    img = im2double(imresize(imread(fullfile(valDir,valFiles(i).name)), imgSize));
    X_val(:,i) = reshape(img, [], 1);
end
for i = 1:numTest
    img = im2double(imresize(imread(fullfile(testDir,testFiles(i).name)), imgSize));
    X_test(:,i) = reshape(img, [], 1);
end

img_dim = size(X_train,1);
z_dim = 100;
hidden_g = 256;
hidden_d = 128;
lr = 0.0005;
epochs = 40;

rng(42);

%% ================================
% 2. Initialize Weights
% ================================
Wg1 = randn(hidden_g, z_dim)*0.02; bg1 = randn(hidden_g,1)*0.01;
Wg2 = randn(img_dim, hidden_g)*0.02; bg2 = randn(img_dim,1)*0.01;
Wd1 = randn(hidden_d, img_dim)*0.02; bd1 = randn(hidden_d,1)*0.01;
Wd2 = randn(1, hidden_d)*0.02; bd2 = randn(1,1)*0.01;

lossD = zeros(epochs,1);
lossG = zeros(epochs,1);

%% ================================
% 3. Training Loop
% ================================
for epoch = 1:epochs
    z = randn(z_dim,1);
    h_g1 = max(0, Wg1*z + bg1);
    img_fake = tanh(Wg2*h_g1 + bg2);
    
    idx = randi(numTrain);
    x_real = X_train(:,idx);
    
    h_d1_real = max(0, Wd1*x_real + bd1);
    h_d1_fake = max(0, Wd1*img_fake + bd1);
    out_real = 1 ./ (1 + exp(-(Wd2*h_d1_real + bd2)));
    out_fake = 1 ./ (1 + exp(-(Wd2*h_d1_fake + bd2)));
    
    loss_D = -mean(log(out_real) + log(1 - out_fake));
    loss_G = -mean(log(out_fake));
    lossD(epoch) = loss_D;
    lossG(epoch) = loss_G;
    
    % Backpropagation (simple gradient steps)
    grad_out_fake = out_fake - 1;
    grad_Wd2 = grad_out_fake * h_d1_fake';
    grad_bd2 = sum(grad_out_fake);
    grad_h_d1_fake = (Wd2' * grad_out_fake) .* (h_d1_fake>0);
    grad_Wd1 = grad_h_d1_fake * img_fake';
    grad_bd1 = sum(grad_h_d1_fake,2);
    
    Wd2 = Wd2 - lr*grad_Wd2; bd2 = bd2 - lr*grad_bd2;
    Wd1 = Wd1 - lr*grad_Wd1; bd1 = bd1 - lr*grad_bd1;
    
    grad_fake_to_D = (Wd1' * grad_h_d1_fake);
    grad_h_g2 = grad_fake_to_D .* (1 - img_fake.^2);
    grad_Wg2 = grad_h_g2 * h_g1';
    grad_bg2 = sum(grad_h_g2,2);
    grad_h_g1 = (Wg2' * grad_h_g2) .* (h_g1>0);
    grad_Wg1 = grad_h_g1 * z';
    grad_bg1 = sum(grad_h_g1,2);
    
    Wg2 = Wg2 - lr*grad_Wg2; bg2 = bg2 - lr*grad_bg2;
    Wg1 = Wg1 - lr*grad_Wg1; bg1 = bg1 - lr*grad_bg1;
    
    if mod(epoch,10)==0
        subplot(1,2,1);
        imshow((reshape(img_fake,[64,64,3])+1)/2);
        title(sprintf('Generated Face (Epoch %d)', epoch));
        subplot(1,2,2);
        plot(lossD(1:epoch),'r','LineWidth',1.5); hold on;
        plot(lossG(1:epoch),'b','LineWidth',1.5); hold off;
        legend('D Loss','G Loss'); title('Training Progress');
        drawnow;
    end
end

%% ================================
% 4. Validation & Testing
% ================================
z_val = randn(z_dim,1);
val_face = tanh(Wg2*max(0,Wg1*z_val+bg1)+bg2);
imshow((reshape(val_face,[64,64,3])+1)/2);
title('Generated Validation Face');

z_test = randn(z_dim,1);
test_face = tanh(Wg2*max(0,Wg1*z_test+bg1)+bg2);
imshow((reshape(test_face,[64,64,3])+1)/2);
title('Generated Testing Face');

%% ================================
% 5. Hardware Fixed-Point Simulation (Q7.8)
% ================================
WL = 16; FL = 8;
T = numerictype(1,WL,FL);
F = fimath('RoundingMethod','Nearest','OverflowAction','Saturate');

z_fx = fi(randn(z_dim,1),T,F);
Wg1_fx = fi(Wg1,T,F); bg1_fx = fi(bg1,T,F);
Wg2_fx = fi(Wg2,T,F); bg2_fx = fi(bg2,T,F);

h_g1_fx = fi(Wg1_fx*z_fx + bg1_fx,T,F); h_g1_fx(h_g1_fx<0)=fi(0,T,F);
img_fx = fi(Wg2_fx*h_g1_fx + bg2_fx,T,F);
img_fx = fi(max(-1,min(1,double(img_fx))),T,F);

figure;
imshow((reshape(double(img_fx),[64,64,3])+1)/2);
title('Generated Face (Fixed Point Hardware)');

%% ================================
% 6. Metrics & Clock Estimation
% ================================
mse_gen = mean((double(val_face)-double(img_fx)).^2);
error_pct = sqrt(mse_gen)*100;

ops_total = numel(Wg1)+numel(Wg2)+numel(Wd1)+numel(Wd2);
clk_freq = 100e6;
clk_cycles = ops_total * epochs;
est_hw_time = clk_cycles / clk_freq;

fprintf('\n=== GAN Performance Summary ===\n');
fprintf('MSE (Float vs Fixed)     : %.6f\n', mse_gen);
fprintf('Error (%%)                : %.3f %%\n', error_pct);
fprintf('Total Ops (all epochs)   : %.0f\n', clk_cycles);
fprintf('Est. HW Time @100MHz     : %.6f s\n', est_hw_time);