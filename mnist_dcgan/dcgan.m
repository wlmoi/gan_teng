% === SKRIP MATLAB LENGKAP DENGAN PERBAIKAN SINTAKS ===

% Asumsi parameter dari arsitektur DCGAN standar
nz = 100;  % Ukuran vektor noise (input)
ngf = 64;  % Ukuran fitur generator
nc = 1;    % Jumlah channel output (1 utk grayscale)
ndf = 64;  % Ukuran fitur diskriminator

%% 1. Definisi Arsitektur Generator (G)
% Arsitektur ini adalah asumsi. HARUS SAMA DENGAN dcgan.py
% Ini adalah arsitektur umum untuk menghasilkan gambar 64x64

layersG = [
    % Menggunakan imageInputLayer untuk input noise [1x1xnz]
    imageInputLayer([1 1 nz], 'Name', 'noise', 'Normalization', 'none')
    
    % Input: 1x1xnz -> Output: 4x4x(ngf*8)
    transposedConv2dLayer(4, ngf * 8, 'Stride', 1, 'Cropping', 0, 'Name', 'tconv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    
    % Input: 4x4x(ngf*8) -> Output: 8x8x(ngf*4)
    transposedConv2dLayer(4, ngf * 4, 'Stride', 2, 'Cropping', 1, 'Name', 'tconv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    
    % Input: 8x8x(ngf*4) -> Output: 16x16x(ngf*2)
    transposedConv2dLayer(4, ngf * 2, 'Stride', 2, 'Cropping', 1, 'Name', 'tconv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    
    % Input: 16x16x(ngf*2) -> Output: 32x32xngf
    transposedConv2dLayer(4, ngf, 'Stride', 2, 'Cropping', 1, 'Name', 'tconv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    
    % Input: 32x32xngf -> Output: 64x64xnc
    transposedConv2dLayer(4, nc, 'Stride', 2, 'Cropping', 1, 'Name', 'tconv5')
    tanhLayer('Name', 'tanh_out') % PyTorch DCGAN biasanya pakai Tanh
];

lgraphG = layerGraph(layersG);
netG = dlnetwork(lgraphG);

%% 2. Definisi Arsitektur Diskriminator (D)
% Asumsi input gambar 64x64

layersD = [
    imageInputLayer([64 64 nc], 'Name', 'image', 'Normalization', 'none')

    % Input: 64x64xnc -> Output: 32x32xndf
    convolution2dLayer(4, ndf, 'Stride', 2, 'Padding', 1, 'Name', 'conv1')
    leakyReluLayer(0.2, 'Name', 'lrelu1') % DCGAN pakai LeakyReLU

    % Input: 32x32xndf -> Output: 16x16x(ndf*2)
    convolution2dLayer(4, ndf * 2, 'Stride', 2, 'Padding', 1, 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    leakyReluLayer(0.2, 'Name', 'lrelu2')

    % Input: 16x16x(ndf*2) -> Output: 8x8x(ndf*4)
    convolution2dLayer(4, ndf * 4, 'Stride', 2, 'Padding', 1, 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    leakyReluLayer(0.2, 'Name', 'lrelu3')

    % Input: 8x8x(ndf*4) -> Output: 4x4x(ndf*8)
    convolution2dLayer(4, ndf * 8, 'Stride', 2, 'Padding', 1, 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    leakyReluLayer(0.2, 'Name', 'lrelu4')

    % Input: 4x4x(ndf*8) -> Output: 1x1x1
    convolution2dLayer(4, 1, 'Stride', 1, 'Padding', 0, 'Name', 'conv5')
    sigmoidLayer('Name', 'sigmoid_out') % Output probabilitas
];

lgraphD = layerGraph(layersD);
netD = dlnetwork(lgraphD);

disp("Arsitektur model Generator dan Diskriminator berhasil dibuat di MATLAB.");

%% 3. Memuat Bobot dari File .mat
% PERINGATAN: Bagian ini masih bersifat ASUMSI dan kemungkinan
% besar akan menimbulkan error. Nama variabel (cth: 'main_0_weight')
% HARUS SAMA PERSIS dengan nama yang diekspor dari Python.

try
    disp("Memuat bobot Generator dari netG_weights.mat...");
    weightsG = load('netG_weights.mat');
    
    % PERHATIAN: Mapping ini HARUS sesuai dengan urutan layer di 'dcgan.py'
    % PyTorch biasanya menamai layer 'main.0.weight', 'main.1.weight', dst.
    % Skrip Python saya mengganti '.' menjadi '_'
    
    % Contoh mapping (ASUMSI NAMA DARI PYTORCH):
    % Asumsi 'main.0' adalah tconv1
    netG.Layers(2).Weights = weightsG.main_0_weight;
    netG.Layers(2).Bias = weightsG.main_0_bias;
    % Asumsi 'main.1' adalah bn1
    netG.Layers(3).Scale = weightsG.main_1_weight; % BN di PyTorch punya 'weight' (scale) dan 'bias'
    netG.Layers(3).Offset = weightsG.main_1_bias;
    netG.Layers(3).TrainedMean = weightsG.main_1_running_mean;
    netG.Layers(3).TrainedVariance = weightsG.main_1_running_var;
    
    % Asumsi 'main.3' adalah tconv2
    netG.Layers(5).Weights = weightsG.main_3_weight;
    netG.Layers(5).Bias = weightsG.main_3_bias;
    % Asumsi 'main.4' adalah bn2
    netG.Layers(6).Scale = weightsG.main_4_weight;
    netG.Layers(6).Offset = weightsG.main_4_bias;
    netG.Layers(6).TrainedMean = weightsG.main_4_running_mean;
    netG.Layers(6).TredVariance = weightsG.main_4_running_var;
    
    % ... Anda harus melanjutkan mapping ini untuk SEMUA layer ...
    % (tconv3, bn3, tconv4, bn4)
    
    % ========================================================
    % --- PERBAIKAN SINTAKS ADA DI BARIS BERIKUTNYA ---
    % ========================================================
    
    % Mapping layer terakhir (tconv5, asumsi 'main.9')
    % Indeks layer (14) mungkin perlu disesuaikan
    netG.Layers(14).Weights = weightsG.main_9_weight;
    netG.Layers(14).Bias = weightsG.main_9_bias; % <-- Ini baris yang diperbaiki (sebelumnya '1Asums4')

    disp("Bobot Generator berhasil dimuat (secara parsial).");
    
catch e
    disp("ERROR saat memuat bobot Generator:");
    disp("Pastikan 'netG_weights.mat' ada di folder yang sama.");
    disp("Pastikan juga nama struct (cth: 'main_0_weight') sudah benar.");
    disp("Jika arsitektur Anda berbeda, Anda perlu menyesuaikan mapping bobot manual.");
    disp(e.message);
end

% Ulangi proses yang sama untuk Diskriminator (netD)
% ... (load('netD_weights.mat'), lalu petakan bobotnya) ...


%% 4. Menghasilkan Gambar (Sesuai Cell 6 di Notebook Anda)

disp("Menghasilkan gambar dari Generator...");

% Buat noise acak
numImages = 16;
% Menyesuaikan dimensi noise agar [H, W, C, B] -> [1, 1, 100, 16]
noise = randn(1, 1, nz, numImages, 'single');

% Konversi ke dlarray
noise = dlarray(noise, 'SSCB'); % Spatial, Spatial, Channel, Batch

% Pindahkan ke GPU jika ada
if canUseGPU
    disp("Menggunakan GPU...");
    noise = gpuArray(noise);
    netG = dlupdate(netG, @gpuArray);
end

% Hasilkan gambar
generatedImages_dl = predict(netG, noise);

% Ambil data dari GPU (jika perlu) dan dlarray
generatedImages = gather(extractdata(generatedImages_dl));

% Tampilkan gambar (mirip dengan matplotlib)
figure;
montage(generatedImages);
title("Gambar yang Dihasilkan oleh Generator (MATLAB)");

%% 5. Menjalankan Diskriminator (Sesuai Cell 8 di Notebook Anda)
% Kode ini hanya akan berfungsi jika Anda sudah memuat bobot netD

% scores = predict(netD, generatedImages_dl);
% disp("Skor dari Diskriminator:");
% disp(gather(extractdata(scores)));