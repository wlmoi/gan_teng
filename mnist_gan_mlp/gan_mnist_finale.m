% Skrip MATLAB untuk menampilkan gambar ASLI vs PALSU
% Dirancang untuk sampel kecil dan perbandingan visual
clear; clc; close all;

% --- Parameter ---
batch_size = 8;     % Gunakan sampel kecil, misal 8
latent_size = 64;

%% 1. Impor dan Inisialisasi Model
disp("Mengimpor dan inisialisasi model...");

% Impor Generator
G_net = importNetworkFromONNX('G.onnx');
G_net = initialize(G_net, dlarray(randn(latent_size, 1, 'single'), 'CB'));

disp("Model siap.");

%% 2. Buat Gambar PALSU (dari Generator)
disp("Membuat gambar palsu...");

% Buat vektor laten acak BARU setiap kali dijalankan
z = randn(latent_size, batch_size, 'single');
dl_z = dlarray(z, 'CB');

% Jalankan Generator
dl_fake_images_vec = predict(G_net, dl_z); % Output [784 x 8]

% --- Pasca-pemrosesan Gambar Palsu ---
fake_images_vec = extractdata(dl_fake_images_vec);
fake_images_denorm = (fake_images_vec + 1) / 2.0; % Denormalisasi
fake_images_distorted = reshape(fake_images_denorm, 28, 28, 1, batch_size);

% --- PERBAIKAN VISUAL (Row-Major vs Column-Major) ---
% Transpose setiap gambar agar tampilannya benar
fake_images_display = pagetranspose(fake_images_distorted);


%% 3. Muat Gambar ASLI (dari MNIST)
disp("Memuat gambar asli...");

[testImages, ~] = digitTest4DArrayData;
% Ambil 8 gambar asli pertama untuk perbandingan
real_images_display = testImages(:,:,1,1:batch_size);

%% 4. Tampilkan Hasil Berdampingan
disp("Menampilkan gambar...");

figure;
% Atur ukuran jendela agar lebih lebar
set(gcf, 'Position', [100, 100, 800, 400]);

% Plot Gambar Asli
subplot(1, 2, 1);
montage(real_images_display);
title('Gambar ASLI (Real Images)');
axis equal;

% Plot Gambar Palsu
subplot(1, 2, 2);
montage(fake_images_display); % Gunakan variabel yang sudah diperbaiki
title('Gambar PALSU (Fake Images)');
axis equal;