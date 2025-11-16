% === SKRIP MATLAB LENGKAP DENGAN PERBAIKAN RESHAPE BATCHNORM ===

% Asumsi parameter dari arsitektur DCGAN standar
nz = 100;  % Ukuran vektor noise (input)
ngf = 64;  % Ukuran fitur generator
nc = 1;    % Jumlah channel output (1 utk grayscale)
ndf = 64;  % Ukuran fitur diskriminator

%% 1. Definisi Arsitektur Generator (G)
disp("Mendefinisikan arsitektur Generator...");
layersG = [
    imageInputLayer([1 1 nz], 'Name', 'noise', 'Normalization', 'none')
    transposedConv2dLayer(4, ngf * 8, 'Stride', 1, 'Cropping', 0, 'Name', 'tconv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    transposedConv2dLayer(4, ngf * 4, 'Stride', 2, 'Cropping', 1, 'Name', 'tconv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    transposedConv2dLayer(4, ngf * 2, 'Stride', 2, 'Cropping', 1, 'Name', 'tconv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    transposedConv2dLayer(4, ngf, 'Stride', 2, 'Cropping', 1, 'Name', 'tconv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    transposedConv2dLayer(4, nc, 'Stride', 2, 'Cropping', 1, 'Name', 'tconv5')
    tanhLayer('Name', 'tanh_out')
];

%% 2. Definisi Arsitektur Diskriminator (D)
disp("Mendefinisikan arsitektur Diskriminator...");
layersD = [
    imageInputLayer([64 64 nc], 'Name', 'image', 'Normalization', 'none')
    convolution2dLayer(4, ndf, 'Stride', 2, 'Padding', 1, 'Name', 'conv1')
    leakyReluLayer(0.2, 'Name', 'lrelu1')
    convolution2dLayer(4, ndf * 2, 'Stride', 2, 'Padding', 1, 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    leakyReluLayer(0.2, 'Name', 'lrelu2')
    convolution2dLayer(4, ndf * 4, 'Stride', 2, 'Padding', 1, 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    leakyReluLayer(0.2, 'Name', 'lrelu3')
    convolution2dLayer(4, ndf * 8, 'Stride', 2, 'Padding', 1, 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    leakyReluLayer(0.2, 'Name', 'lrelu4')
    convolution2dLayer(4, 1, 'Stride', 1, 'Padding', 0, 'Name', 'conv5')
    sigmoidLayer('Name', 'sigmoid_out')
];

%% 3. Memuat Bobot KE DALAM ARSITEKTUR

% Pengecekan file
if ~exist('netG_weights.mat', 'file')
    error("STOP: File 'netG_weights.mat' TIDAK DITEMUKAN. Pastikan file ada di 'Current Folder' MATLAB (D:\GANTeng\mnist_dcgan).");
end
if ~exist('netD_weights.mat', 'file')
    error("STOP: File 'netD_weights.mat' TIDAK DITEMUKAN. Pastikan file ada di 'Current Folder' MATLAB (D:\GANTeng\mnist_dcgan).");
end
disp("Pengecekan file berhasil. 'netG_weights.mat' dan 'netD_weights.mat' ditemukan.");

% --- GENERATOR (G) ---
try
    disp("Memuat bobot Generator ke arsitektur...");
    weightsG = load('netG_weights.mat');
    
    % =======================================================================
    % === PERBAIKAN: Menggunakan reshape(param, [1, 1, []]) ===
    % Ini memaksa MATLAB membuat array [1, 1, C] tidak peduli
    % inputnya [1, C] (baris) atau [C, 1] (kolom).
    % =======================================================================
    reshapeBN = @(param) reshape(param, [1, 1, []]);

    % Permute [3, 4, 2, 1] sudah benar (dari error sebelumnya)
    % PyTorch (G): [in, out, kH, kW] -> [100, 512, 4, 4]
    % MATLAB (G): [kH, kW, out, in] -> [4, 4, 512, 100]
    
    % Blok 1 (main.0 = tconv1, main.1 = bn1)
    layersG(2).Weights = permute(weightsG.main_0_weight, [3, 4, 2, 1]);
    layersG(3).Scale = reshapeBN(weightsG.main_1_weight);
    layersG(3).Offset = reshapeBN(weightsG.main_1_bias);
    layersG(3).TrainedMean = reshapeBN(weightsG.main_1_running_mean);
    layersG(3).TrainedVariance = reshapeBN(weightsG.main_1_running_var);
    
    % Blok 2 (main.3 = tconv2, main.4 = bn2)
    layersG(5).Weights = permute(weightsG.main_3_weight, [3, 4, 2, 1]);
    layersG(6).Scale = reshapeBN(weightsG.main_4_weight);
    layersG(6).Offset = reshapeBN(weightsG.main_4_bias);
    layersG(6).TrainedMean = reshapeBN(weightsG.main_4_running_mean);
    layersG(6).TrainedVariance = reshapeBN(weightsG.main_4_running_var);

    % Blok 3 (main.6 = tconv3, main.7 = bn3)
    layersG(8).Weights = permute(weightsG.main_6_weight, [3, 4, 2, 1]);
    layersG(9).Scale = reshapeBN(weightsG.main_7_weight);
    layersG(9).Offset = reshapeBN(weightsG.main_7_bias);
    layersG(9).TrainedMean = reshapeBN(weightsG.main_7_running_mean);
    layersG(9).TrainedVariance = reshapeBN(weightsG.main_7_running_var);

    % Blok 4 (main.9 = tconv4, main.10 = bn4)
    layersG(11).Weights = permute(weightsG.main_9_weight, [3, 4, 2, 1]);
    layersG(12).Scale = reshapeBN(weightsG.main_10_weight);
    layersG(12).Offset = reshapeBN(weightsG.main_10_bias);
    layersG(12).TrainedMean = reshapeBN(weightsG.main_10_running_mean);
    layersG(12).TrainedVariance = reshapeBN(weightsG.main_10_running_var);
    
    % Blok 5 (main.12 = tconv5)
    layersG(14).Weights = permute(weightsG.main_12_weight, [3, 4, 2, 1]);

    disp("Bobot Generator berhasil DI-INJEK ke arsitektur.");
    
catch e
    disp("ERROR SAAT MEMASUKKAN BOBOT GENERATOR:");
    disp(e.message);
    error("Gagal memuat bobot G. Cek nama field di .mat (cth: 'main_0_weight')");
end

% --- DISCRIMINATOR (D) ---
try
    disp("Memuat bobot Diskriminator ke arsitektur...");
    weightsD = load('netD_weights.mat');
    
    % Definisikan ulang helper function untuk blok ini
    reshapeBN = @(param) reshape(param, [1, 1, []]);
    
    % Permute [3, 4, 2, 1] sudah benar
    % PyTorch (D): [out, in, kH, kW] -> [64, 1, 4, 4]
    % MATLAB (D): [kH, kW, in, out] -> [4, 4, 1, 64]
    
    % Blok 1 (main.0 = conv1)
    layersD(2).Weights = permute(weightsD.main_0_weight, [3, 4, 2, 1]);
    
    % Blok 2 (main.2 = conv2, main.3 = bn2)
    layersD(4).Weights = permute(weightsD.main_2_weight, [3, 4, 2, 1]);
    layersD(5).Scale = reshapeBN(weightsD.main_3_weight);
    layersD(5).Offset = reshapeBN(weightsD.main_3_bias);
    layersD(5).TrainedMean = reshapeBN(weightsD.main_3_running_mean);
    layersD(5).TrainedVariance = reshapeBN(weightsD.main_3_running_var);

    % Blok 3 (main.5 = conv3, main.6 = bn3)
    layersD(7).Weights = permute(weightsD.main_5_weight, [3, 4, 2, 1]);
    layersD(8).Scale = reshapeBN(weightsD.main_6_weight);
    layersD(8).Offset = reshapeBN(weightsD.main_6_bias);
    layersD(8).TrainedMean = reshapeBN(weightsD.main_6_running_mean);
    layersD(8).TrainedVariance = reshapeBN(weightsD.main_6_running_var);
    
    % Blok 4 (main.8 = conv4, main.9 = bn4)
    layersD(10).Weights = permute(weightsD.main_8_weight, [3, 4, 2, 1]);
    layersD(11).Scale = reshapeBN(weightsD.main_9_weight);
    layersD(11).Offset = reshapeBN(weightsD.main_9_bias);
    layersD(11).TrainedMean = reshapeBN(weightsD.main_9_running_mean);
    layersD(11).TrainedVariance = reshapeBN(weightsD.main_9_running_var);
    
    % Blok 5 (main.11 = conv5)
    layersD(13).Weights = permute(weightsD.main_11_weight, [3, 4, 2, 1]);

    disp("Bobot Diskriminator berhasil DI-INJEK ke arsitektur.");
    
catch e
    disp("ERROR SAAT MEMASUKKAN BOBOT DISKRIMINATOR:");
    disp(e.message);
    error("Gagal memuat bobot D. Cek nama field di .mat (cth: 'main_0_weight')");
end


%% 4. Merakit Jaringan (SETELAH BOBOT DIMUAT)
disp("Merakit arsitektur Generator yang sudah berisi bobot...");
lgraphG = layerGraph(layersG);
netG = dlnetwork(lgraphG);

disp("Merakit arsitektur Diskriminator yang sudah berisi bobot...");
lgraphD = layerGraph(layersD);
netD = dlnetwork(lgraphD);

disp("Arsitektur model Generator dan Diskriminator berhasil dibuat di MATLAB.");


%% 5. Menghasilkan Gambar (Sesuai Cell 6 di Notebook Anda)
disp("Menghasilkan gambar dari Generator...");
numImages = 16;
noise = randn(1, 1, nz, numImages, 'single');
noise = dlarray(noise, 'SSCB');

if canUseGPU
    disp("Menggunakan GPU...");
    noise = gpuArray(noise);
    netG = dlupdate(netG, @gpuArray);
end

generatedImages_dl = predict(netG, noise);
generatedImages = gather(extractdata(generatedImages_dl));

figure;
montage(generatedImages);
title("Gambar yang Dihasilkan oleh Generator (MATLAB)");

%% 6. Menjalankan Diskriminator (Sesuai Cell 8 di Notebook Anda)
disp("Menjalankan Diskriminator pada gambar yang dihasilkan...");
if canUseGPU
    netD = dlupdate(netD, @gpuArray);
end

scores = predict(netD, generatedImages_dl);
disp("Skor dari Diskriminator:");
scores_vector = squeeze(gather(extractdata(scores)));
disp(scores_vector');