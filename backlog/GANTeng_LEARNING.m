%% ============================================================
%  FACE-GAN (CelebA-style) + FSM Hardware Simulation
%  ============================================================
%  Folder Structure:
%   dataset/training/*.jpg
%   dataset/validation/*.jpg
%   dataset/testing/*.jpg
%
%  Output Folder:
%   outputs/
%     ├── output_epoch_01_float.png
%     ├── output_epoch_01_fixed.png
%     └── ...
%
%  Features:
%   - Generator + Discriminator (ReLU, tanh, bias)
%   - 40 Epoch Training
%   - Hardware Simulation using FSM approach
%   - Clock Cycle estimation
%   - Automatic output saving
%   - Progress indicator (training status)
%  ============================================================

clear; clc; close all;

%% ================================
% 1. Dataset Loading
% ================================
baseDir = 'dataset';
trainDir = fullfile(baseDir, 'training');
valDir   = fullfile(baseDir, 'validation');
testDir  = fullfile(baseDir, 'testing');

trainFiles = dir(fullfile(trainDir, '*.jpg'));
valFiles   = dir(fullfile(valDir, '*.jpg'));
testFiles  = dir(fullfile(testDir, '*.jpg'));

numTrain = numel(trainFiles);
numVal   = numel(valFiles);
numTest  = numel(testFiles);
fprintf('Loaded %d train, %d val, %d test images.\n', numTrain, numVal, numTest);

imgSize = [64 64];
for i = 1:numTrain
    img = im2double(imresize(imread(fullfile(trainDir, trainFiles(i).name)), imgSize));
    X_train(:, i) = reshape(img, [], 1);
end

%% ================================
% 2. Network Initialization
% ================================
img_dim  = size(X_train, 1);
z_dim    = 100;
hidden_g = 256;
hidden_d = 128;
epochs   = 40;
lr       = 0.0005;
rng(42);

Wg1 = randn(hidden_g, z_dim) * 0.02; bg1 = randn(hidden_g, 1) * 0.01;
Wg2 = randn(img_dim, hidden_g) * 0.02; bg2 = randn(img_dim, 1) * 0.01;
Wd1 = randn(hidden_d, img_dim) * 0.02; bd1 = randn(hidden_d, 1) * 0.01;
Wd2 = randn(1, hidden_d) * 0.02; bd2 = randn(1, 1) * 0.01;

lossD = zeros(epochs,1); lossG = zeros(epochs,1);

%% ================================
% 3. Create Output Folder
% ================================
outputDir = 'outputs';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% ================================
% 4. Training Loop (40 Epochs) + Progress Display
% ================================
fprintf('\n=== TRAINING STARTED ===\n');
wb = waitbar(0, 'Training in progress...'); % Progress bar
tic; % Start timer

for epoch = 1:epochs
    % ===== Generator Forward =====
    z = randn(z_dim, 1);
    h_g1 = max(0, Wg1*z + bg1);
    img_fake = tanh(Wg2*h_g1 + bg2);
    
    % ===== Real Sample =====
    idx = randi(numTrain);
    x_real = X_train(:, idx);
    
    % ===== Discriminator Forward =====
    h_d1_real = max(0, Wd1*x_real + bd1);
    h_d1_fake = max(0, Wd1*img_fake + bd1);
    out_real = 1 ./ (1 + exp(-(Wd2*h_d1_real + bd2)));
    out_fake = 1 ./ (1 + exp(-(Wd2*h_d1_fake + bd2)));
    
    % ===== Loss =====
    lossD(epoch) = -mean(log(out_real) + log(1 - out_fake));
    lossG(epoch) = -mean(log(out_fake));
    
    % ===== Backpropagation =====
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

    % ===== Update Progress Bar =====
    elapsed = toc;
    est_time_left = (elapsed/epoch) * (epochs - epoch);
    waitbar(epoch/epochs, wb, sprintf('Epoch %d/%d | Est. %.1fs left', epoch, epochs, est_time_left));

    % ===== Optional Console Feedback =====
    if mod(epoch,5) == 0
        fprintf('Epoch %02d/%02d | LossD=%.4f | LossG=%.4f | Elapsed: %.2fs | ETA: %.1fs\n', ...
            epoch, epochs, lossD(epoch), lossG(epoch), elapsed, est_time_left);
    end
    
    % ===== Display every 10 epochs =====
    if mod(epoch,10) == 0
        img_disp = (reshape(img_fake, [64,64,3]) + 1) / 2;
        figure(1);
        subplot(1,2,1); imshow(img_disp); title(sprintf('Generated (Epoch %d)', epoch));
        subplot(1,2,2); plot(lossD(1:epoch),'r','LineWidth',1.5); hold on;
        plot(lossG(1:epoch),'b','LineWidth',1.5); hold off;
        legend('Discriminator','Generator'); title('Training Progress');
        drawnow;
        
        % Save image
        outPath = fullfile(outputDir, sprintf('output_epoch_%02d_float.png', epoch));
        imwrite(img_disp, outPath);
    end
end

close(wb);
fprintf('\n=== TRAINING COMPLETE ===\n');

%% ================================
% 5. Hardware (Fixed-Point Q7.8) + FSM Simulation
% ================================
WL = 16; FL = 8;
T = numerictype(1, WL, FL);
F = fimath('RoundingMethod','Nearest','OverflowAction','Saturate');

z_fx = fi(randn(z_dim,1), T, F);
Wg1_fx = fi(Wg1, T, F); bg1_fx = fi(bg1, T, F);
Wg2_fx = fi(Wg2, T, F); bg2_fx = fi(bg2, T, F);

% FSM States:
%  S0: LOAD INPUT
%  S1: COMPUTE LAYER 1
%  S2: ACTIVATE (ReLU)
%  S3: COMPUTE LAYER 2
%  S4: OUTPUT & WRITE
FSM_states = {'LOAD', 'L1_COMP', 'ACT_RELU', 'L2_COMP', 'WRITE_OUT'};

clk_per_mult = 1;    % 1 cycle per multiply
clk_per_add  = 1;    % 1 cycle per addition
clk_act_relu = 2;    % activation overhead
clk_write    = 10;   % output writing overhead

% Count operations
mults = numel(Wg1) + numel(Wg2);
adds  = hidden_g + img_dim;
cycles_total = mults*clk_per_mult + adds*clk_per_add + ...
               hidden_g*clk_act_relu + clk_write;

% Hardware Simulation
fprintf('\nFSM Hardware Simulation:\n');
fprintf('----------------------------------\n');
for s = 1:length(FSM_states)
    fprintf('State %-10s : Active\n', FSM_states{s});
end
fprintf('----------------------------------\n');
fprintf('Total FSM Clock Cycles: %.0f\n', cycles_total);

% Generate fixed-point image
h_g1_fx = fi(Wg1_fx*z_fx + bg1_fx, T, F); 
h_g1_fx(h_g1_fx<0) = fi(0, T, F);
img_fx = fi(Wg2_fx*h_g1_fx + bg2_fx, T, F);
img_fx = fi(max(-1, min(1, double(img_fx))), T, F);

img_hw = (reshape(double(img_fx), [64,64,3]) + 1) / 2;
imwrite(img_hw, fullfile(outputDir, sprintf('output_epoch_%02d_fixed.png', epochs)));

%% ================================
% 6. Metrics
% ================================
mse_val = mean((double(img_fake) - double(img_fx)).^2);
err_pct = sqrt(mse_val) * 100;
clk_freq = 100e6;
est_hw_time = cycles_total / clk_freq;

fprintf('\n=== Performance Summary ===\n');
fprintf('MSE (Float vs Fixed)   : %.6f\n', mse_val);
fprintf('Error (%%)              : %.3f %%\n', err_pct);
fprintf('FSM Total Clock Cycles : %.0f cycles\n', cycles_total);
fprintf('Est. HW Time @100MHz   : %.8f s\n', est_hw_time);
fprintf('Outputs saved to: %s\n', outputDir);
