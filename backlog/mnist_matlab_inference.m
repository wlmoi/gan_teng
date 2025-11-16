function mnist_matlab_inference()
    % --- FUNGSI UTAMA ---
    
    % NAMA FILE MODEL (bobot) ANDA
    % Ini harus 'mnist_model.mat'
    model_name = 'mnist_model.mat'; 
    
    iterations = 10000;             % Jumlah gambar uji (maks 10000)
    
    fprintf('Memulai inferensi fixed-point...\n');
    forward(model_name, iterations);
end

function forward(model_name, iterations)
    % Fungsi ini menjalankan forward pass manual menggunakan aritmatika fixed-point.
    
    % 1. Muat dan proses data uji
    %    (Fungsi clean_data_test_only dipanggil di bawah)
    [X_test_clean, Y_test_clean] = clean_data_test_only();
    fprintf('Data uji dimuat dan diproses.\n');

    % 2. Muat model (bobot) dari file .mat
    try
        model_data = load(model_name);
    catch
        error('Gagal memuat file model: %s. Pastikan file ada di path MATLAB.', model_name);
    end
    
    % Menggunakan nama variabel yang BENAR dari screenshot Anda
    try
        weights1_f = model_data.layer_0_dense_weights;
        biases1_f = model_data.layer_0_dense_bias;  % <- BENAR (tanpa 's')
        weights2_f = model_data.layer_2_dense_weights;
        biases2_f = model_data.layer_2_dense_bias;  % <- BENAR (tanpa 's')
    catch ME
        error('File .mat tidak berisi variabel bobot yang diharapkan. Error: %s', ME.message);
    end
    
    % Pastikan bias adalah vektor kolom
    biases1_f = biases1_f(:); % (10, 1)
    biases2_f = biases2_f(:); % (10, 1)

    % 3. Konversi bobot ke fixed-point (int8)
    bits_past_radix = 2;
    scale = 2^bits_past_radix;
    
    weights1 = int8(round(weights1_f * scale)); % (784, 10) int8
    biases1 = int8(round(biases1_f * scale));   % (10, 1) int8
    weights2 = int8(round(weights2_f * scale)); % (10, 10) int8
    biases2 = int8(round(biases2_f * scale));   % (10, 1) int8
    
    fprintf('Bobot model dikonversi ke int8 fixed-point.\n');

    % 4. Mulai loop inferensi
    count = 0;
    total = 0;
    
    % Transpose bobot sekali untuk mencocokkan loop Python (akses baris)
    weights1_T = weights1'; % (10, 784) int8
    weights2_T = weights2'; % (10, 10) int8
    
    num_test_images = size(X_test_clean, 1);
    if nargin < 2
        iterations = num_test_images;
    end
    iterations = min(iterations, num_test_images);
    fprintf('Menjalankan inferensi pada %d gambar...\n', iterations);
    
    for i = 1:iterations
        X = X_test_clean(i, :); % (1 x 784) int8 (input biner)
        Y = Y_test_clean(i, :); % (1 x 10) logical (label one-hot)
        
        % --- HIDDEN LAYER (Input -> Dense -> ReLU) ---
        output_hidden = zeros(1, 10, 'int32');
        
        for neuron = 1:10
            weights_neuron = weights1_T(neuron, :); % (1 x 784) int8
            
            weight_acc = int32(0);
            indices = find(X == 1); 
            if ~isempty(indices)
                weight_acc = sum(int32(weights_neuron(indices))); 
            end
            
            check_overflow(weight_acc, 8); 
            weight_acc = weight_acc + int32(biases1(neuron));
            check_overflow(weight_acc, 8); 
            
            output_hidden(neuron) = weight_acc;
        end
        
        hidden_out = int8(output_hidden); 

        % --- RELU ACTIVATION ---
        hidden_out = max(0, hidden_out); % (1 x 10) int8
        
        % --- OUTPUT LAYER (ReLU -> Dense -> ArgMax) ---
        output_final = zeros(1, 10, 'int32');
        for neuron = 1:10
            weights_neuron = weights2_T(neuron, :); % (1 x 10) int8
            
            weight_acc = int32(0);
            for index = 1:10
                prod = int32(weights_neuron(index)) * int32(hidden_out(index));
                weight_acc = weight_acc + prod;
                check_overflow(weight_acc, 16);
            end
            
            weight_acc = weight_acc + int32(biases2(neuron));
            check_overflow(weight_acc, 16);
            
            output_final(neuron) = weight_acc;
        end
        
        output_out = int16(output_final); 
        
        % --- PREDICTION (ArgMax) ---
        [~, max_idx] = max(output_out); 
        prediction = zeros(1, 10, 'logical');
        prediction(max_idx) = true; 
        
        % --- Compare ---
        if all(prediction == Y) 
            count = count + 1;
        end
        total = total + 1;
        
        if mod(total, 1000) == 0 
            fprintf('... %d gambar diproses\n', total);
        end
    end
    
    % --- Tampilkan Hasil ---
    fprintf('Selesai.\n');
    fprintf('Accuracy: %d / %d = %.2f%%\n', count, total, (count / total) * 100);
end

function [X_test_clean, Y_test_clean] = clean_data_test_only()
    % Fungsi ini memproses data uji MNIST mentah agar sesuai dengan logika Python.
    
    % --- Mulai Bagian yang Harus Diisi Pengguna ---
    try
        % MENCOBA MEMUAT FILE DATA GAMBAR
        data = load('mnist_uint8.mat'); 
        X_test_raw = data.X_test; 
        
        [~, Y_test_raw] = max(data.Y_test, [], 2);
        Y_test_raw = Y_test_raw - 1; % Konversi ke label 0-9
        
    catch
        % JIKA GAGAL, BERI PERINGATAN DAN GUNAKAN DATA ACAK
        warning('Gagal memuat "mnist_uint8.mat". Menggunakan data acak.');
        X_test_raw = rand(10000, 28, 28);
        Y_test_raw = randi([0, 9], 10000, 1);
    end
    % --- Akhir Bagian yang Harus Diisi Pengguna ---

    % Reshape X_test ke (10000, 784)
    if ndims(X_test_raw) == 4 
        X_test_flat = reshape(X_test_raw, 784, 10000)';
    elseif ndims(X_test_raw) == 3 
        X_test_flat = reshape(X_test_raw, 10000, 784);
    else 
        X_test_flat = X_test_raw;
    end

    % Normalisasi (ke 'single' / float32)
    X_test_norm = single(X_test_flat) / 255.0;
    
    % Binarize (sesuai kode Python)
    X_test_clean = int8(X_test_norm >= 0.5); % (10000 x 784) int8
    
    % One-hot encode Y_test
    if size(Y_test_raw, 2) == 1 
        labels = Y_test_raw + 1; 
        num_classes = 10;
        num_samples = length(labels);
        Y_test_clean = logical(sparse(1:num_samples, labels, 1, num_samples, num_classes));
    else 
        Y_test_clean = logical(Y_test_raw);
    end
end

function check_overflow(x, num_bits)
    % Fungsi helper untuk memeriksa overflow
    max_val = 2^(num_bits - 1) - 1;
    min_val = -(2^(num_bits - 1));
    if (x > max_val) || (x < min_val)
        fprintf('PERINGATAN: Overflow terdeteksi! Nilai: %d, Batas: [%d, %d]\n', x, min_val, max_val);
    end
end