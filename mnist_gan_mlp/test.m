% Skrip MATLAB untuk memuat dan menjalankan model GAN (format ONNX)
% Versi final dengan fungsi initialize()
clear; clc; close all;

%% 1. Impor Model dari ONNX
disp("Mengimpor model Diskriminator (D)...");
D_net = importNetworkFromONNX('D.onnx');
% --- PERBAIKAN DI SINI: Inisialisasi jaringan ---
% Kita beri input palsu (dummy) [784x1] berformat 'CB'
disp("Inisialisasi D_net...");
D_net = initialize(D_net, dlarray(randn(784, 1, 'single'), 'CB'));

disp("Mengimpor model Generator (G)...");
G_net = importNetworkFromONNX('G.onnx');
% --- PERBAIKAN DI SINI: Inisialisasi jaringan ---
% Kita beri input palsu (dummy) [64x1] (ukuran laten) berformat 'CB'
disp("Inisialisasi G_net...");
latent_size = 64; % Pastikan ini sama dengan skrip Python
G_net = initialize(G_net, dlarray(randn(latent_size, 1, 'single'), 'CB'));

disp("Impor dan inisialisasi selesai.");

%% 2. Menjalankan Diskriminator (D) pada Gambar Nyata (Real Images)

disp("Memuat dataset MNIST...");
[testImages, ~] = digitTest4DArrayData; 

% Pilih batch gambar, misal 32 gambar pertama
batch_size = 32;
real_images = testImages(:,:,1,1:batch_size);

% --- Pra-pemrosesan Gambar ---
real_images_processed = single(real_images) / 255.0;
real_images_processed = (real_images_processed * 2.0) - 1.0;
real_images_flat = reshape(real_images_processed, 784, batch_size);
dl_real_images = dlarray(real_images_flat, 'CB');

% --- Jalankan Prediksi (Inferensi) ---
disp("Menjalankan Diskriminator pada gambar nyata...");
% Baris ini sekarang seharusnya bekerja
real_scores = predict(D_net, dl_real_images);

% Tampilkan skor
disp("Skor untuk gambar nyata (1 = nyata, 0 = palsu):");
disp(extractdata(real_scores));

%% 3. Menjalankan Generator (G) untuk Membuat Gambar Palsu

disp("Menjalankan Generator untuk membuat gambar palsu...");
% Buat vektor laten acak
z = randn(latent_size, batch_size, 'single');
dl_z = dlarray(z, 'CB');

% Hasilkan gambar palsu
dl_fake_images = predict(G_net, dl_z);

% --- Pasca-pemrosesan Gambar ---
fake_images = extractdata(dl_fake_images);
fake_images = (fake_images + 1) / 2.0; % Denormalisasi
fake_images_grid = reshape(fake_images, 28, 28, 1, batch_size);

% Tampilkan gambar yang dihasilkan dalam satu montase
disp("Menampilkan gambar yang dihasilkan oleh Generator...");
figure;
montage(fake_images_grid, 'Size', [4 8]);
title('Gambar Palsu yang Dihasilkan oleh Generator (di MATLAB)');