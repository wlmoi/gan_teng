% === SKRIP MATLAB LENGKAP DENGAN PERBAIKAN PERMUTE DIMENSI ===

% Asumsi parameter dari arsitektur DCGAN standar
nz = 100;  % Ukuran vektor noise (input)
ngf = 64;  % Ukuran fitur generator
nc = 1;    % Jumlah channel output (1 utk grayscale)
ndf = 64;  % Ukuran fitur diskriminator

%% 1. Definisi Arsitektur Generator (G)
% Arsitektur ini SUDAH SESUAI dengan gambar 'Generator.png'

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

lgraphG = layerGraph(layersG);
netG = dlnetwork(lgraphG);

%% 2. Definisi Arsitektur Diskriminator (D)
% Arsitektur ini SUDAH SESUAI dengan gambar 'Discriminator.png'

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

lgraphD = layerGraph(layersD);
netD = dlnetwork(lgraphD);

disp("Arsitektur model Generator dan Diskriminator berhasil dibuat di MATLAB.");

%% 3. Memuat Bobot dari File .mat
% =======================================================================
% === PERBAIKAN UTAMA: Menggunakan permute() untuk bobot konvolusi ===
% Urutan PyTorch [out, in, kH, kW] atau [in, out, kH, kW]
% Urutan MATLAB [kH, kW, in, out] atau [kH, kW, out, in]
% Permutasi yang diperlukan: [3, 4, 2, 1]
% =======================================================================

% --- GENERATOR (G) ---
try
    disp("Memuat bobot Generator dari netG_weights.mat...");
    weightsG = load('netG_weights.mat');
    
    % Blok 1 (main.0 = tconv1, main.1 = bn1)
    netG.Layers(2).Weights = permute(weightsG.main_0_weight, [3, 4, 2, 1]); % TransposeConv [kH, kW, out, in]
    netG.Layers(3).Scale = weightsG.main_1_weight;
    netG.Layers(3).Offset = weightsG.main_1_bias;
    netG.Layers(3).TrainedMean = weightsG.main_1_running_mean;
    netG.Layers(3).TrainedVariance = weightsG.main_1_running_var;
    
    % Blok 2 (main.3 = tconv2, main.4 = bn2)
    netG.Layers(5).Weights = permute(weightsG.main_3_weight, [3, 4, 2, 1]);
    netG.Layers(6).Scale = weightsG.main_4_weight;
    netG.Layers(6).Offset = weightsG.main_4_bias;
    netG.Layers(6).TrainedMean = weightsG.main_4_running_mean;
    netG.Layers(6).TrainedVariance = weightsG.main_4_running_var;

    % Blok 3 (main.6 = tconv3, main.7 = bn3)
    netG.Layers(8).Weights = permute(weightsG.main_6_weight, [3, 4, 2, 1]);
    netG.Layers(9).Scale = weightsG.main_7_weight;
    netG.Layers(9).Offset = weightsG.main_7_bias;
    netG.Layers(9).TrainedMean = weightsG.main_7_running_mean;
    netG.Layers(9).TrainedVariance = weightsG.main_7_running_var;

    % Blok 4 (main.9 = tconv4, main.10 = bn4)
    netG.Layers(11).Weights = permute(weightsG.main_9_weight, [3, 4, 2, 1]);
    netG.Layers(12).Scale = weightsG.main_10_weight;
    netG.Layers(12).Offset = weightsG.main_10_bias;
    netG.Layers(12).TrainedMean = weightsG.main_10_running_mean;
    netG.Layers(12).TrainedVariance = weightsG.main_10_running_var;
    
    % Blok 5 (main.12 = tconv5)
    netG.Layers(14).Weights = permute(weightsG.main_12_weight, [3, 4, 2, 1]);

    disp("Bobot Generator berhasil dimuat.");
    
catch e
    disp("ERROR saat memuat bobot Generator:");
    disp("Pastikan 'netG_weights.mat' ada di folder yang sama.");
    disp("Pastikan Anda sudah menjalankan skrip 'export_weights.py' dari PyTorch.");
    disp(e.message);
end

% --- DISCRIMINATOR (D) ---
try
    disp("Memuat bobot Diskriminator dari netD_weights.mat...");
    weightsD = load('netD_weights.mat');
    
    % Blok 1 (main.0 = conv1)
    netD.Layers(2).Weights = permute(weightsD.main_0_weight, [3, 4, 2, 1]); % Conv [kH, kW, in, out]
    
    % Blok 2 (main.2 = conv2, main.3 = bn2)
    netD.Layers(4).Weights = permute(weightsD.main_2_weight, [3, 4, 2, 1]);
    netD.Layers(5).Scale = weightsD.main_3_weight;
    netD.Layers(5).Offset = weightsD.main_3_bias;
    netD.Layers(5).TrainedMean = weightsD.main_3_running_mean;
    netD.Layers(5).TrainedVariance = weightsD.main_3_running_var;

    % Blok 3 (main.5 = conv3, main.6 = bn3)
    netD.Layers(7).Weights = permute(weightsD.main_5_weight, [3, 4, 2, 1]);
    netD.Layers(8).Scale = weightsD.main_6_weight;
    netD.Layers(8).Offset = weightsD.main_6_bias;
    netD.Layers(8).TrainedMean = weightsD.main_6_running_mean;
    netD.Layers(8).TrainedVariance = weightsD.main_6_running_var;
    
    % Blok 4 (main.8 = conv4, main.9 = bn4)
    netD.Layers(10).Weights = permute(weightsD.main_8_weight, [3, 4, 2, 1]);
    netD.Layers(11).Scale = weightsD.main_9_weight;
    netD.Layers(11).Offset = weightsD.main_9_bias;
    netD.Layers(11).TrainedMean = weightsD.main_9_running_mean;
    netD.Layers(11).TrainedVariance = weightsD.main_9_running_var;
    
    % Blok 5 (main.11 = conv5)
    netD.Layers(13).Weights = permute(weightsD.main_11_weight, [3, 4, 2, 1]);

    disp("Bobot Diskriminator berhasil dimuat.");
    
catch e
    disp("ERROR saat memuat bobot Diskriminator:");
    disp("Pastikan 'netD_weights.mat' ada di folder yang sama.");
    disp("Pastikan Anda sudah menjalankan skrip 'export_weights.py' dari PyTorch.");
    disp(e.message);
end


%% 4. Menghasilkan Gambar (Sesuai Cell 6 di Notebook Anda)

disp("Menghasilkan gambar dari Generator...");

% Buat noise acak
numImages = 16;
% Dimensi [H, W, C, B] -> [1, 1, 100, 16]
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
% Kode ini sekarang seharusnya berfungsi jika bobot netD berhasil dimuat

if canUseGPU
    netD = dlupdate(netD, @gpuArray);
end

scores = predict(netD, generatedImages_dl);
disp("Skor dari Diskriminator:");
% Merapikan output agar mirip dengan Python (opsional)
scores_vector = squeeze(gather(extractdata(scores)));
disp(scores_vector');