% mlp_fixedpoint_sim.m
% MLP forward pass fixed-point simulation (safe + fixed-point modes)
% Usage:
%   - Edit variables loadDataMat = true/false to load your own .mat files:
%       X_test_binary (N x 784), Y_test_onehot (N x 10)
%   - Or leave as dummy random data (example)
% Requirements:
%   - For true fixed-point mode, MATLAB Fixed-Point Designer must be installed.
%   - Otherwise the script runs in float-then-quantize safe mode.

clear; close all; clc;

%% --- User options: change these if you want to load real data ---
loadDataMat = false;          % set true if you have X_test_binary.mat & Y_test_onehot.mat
X_matfile = 'X_test_binary.mat';
Y_matfile = 'Y_test_onehot.mat';
use_fixed_point_if_available = true; % prefer fi if toolbox exists

%% --- Dummy data (replace by loading .mat if desired) ---
rng(0);
weights1_float = randn(784, 10) * 0.1;   % (784 x 10)
biases1_float  = randn(1, 10) * 0.1;     % (1 x 10)
weights2_float = randn(10, 10) * 0.1;    % (10 x 10)
biases2_float  = randn(1, 10) * 0.1;     % (1 x 10)

% single binary sample, one-hot label
X_test_dummy = randi([0, 1], 1, 784);
Y_test_dummy = zeros(1, 10);
Y_test_dummy(randi([1, 10])) = 1;

if loadDataMat
    % Expect variables X_test_binary (N x 784) and Y_test_onehot (N x 10)
    if exist(X_matfile,'file') && exist(Y_matfile,'file')
        Sx = load(X_matfile);
        Sy = load(Y_matfile);
        % adapt field names or variables as loaded
        if isfield(Sx,'X_test_binary')
            X_test = Sx.X_test_binary;
        else
            error('X_test_binary not found in %s', X_matfile);
        end
        if isfield(Sy,'Y_test_onehot')
            Y_test = Sy.Y_test_onehot;
        else
            error('Y_test_onehot not found in %s', Y_matfile);
        end
    else
        error('Data mat files not found. Set loadDataMat = false or provide files.');
    end
else
    X_test = X_test_dummy;
    Y_test = Y_test_dummy;
end

iterations = size(X_test, 1);

%% --- Fixed-point formats (as in your spec) ---
% Hidden: Q5.2 (signed, 8 bits total)
h_wl = 8;  h_fl = 2;

% Output weights: Q2.13 (signed, 16 bits)
o_w_wl = 16; o_w_fl = 13;

% Output bias/output: Q13.2 (signed, 16 bits)
o_b_wl = 16; o_b_fl = 2;

%% --- Check availability of Fixed-Point Designer and choose mode ---
haveFPT = license('test','Fixed_Point_Toolbox') && exist('fi','class');
use_fixed_point = use_fixed_point_if_available && haveFPT;

if use_fixed_point
    fprintf('Running in TRUE fixed-point mode (Fixed-Point Designer available).\n');
    % define fimath objects
    h_fimath = fimath('RoundingMethod','Nearest', 'OverflowAction','Saturate', ...
                      'ProductMode','FullPrecision', 'SumMode','FullPrecision');
    o_w_fimath = fimath('RoundingMethod','Nearest', 'OverflowAction','Saturate', ...
                        'ProductMode','FullPrecision', 'SumMode','FullPrecision');
    o_b_fimath = fimath('RoundingMethod','Nearest', 'OverflowAction','Saturate', ...
                        'ProductMode','FullPrecision', 'SumMode','FullPrecision');

    % Quantize weights and biases to fi objects
    weights1_fi = fi(weights1_float, true, h_wl, h_fl, 'fimath', h_fimath);
    biases1_fi  = fi(biases1_float,  true, h_wl, h_fl, 'fimath', h_fimath);

    weights2_fi = fi(weights2_float, true, o_w_wl, o_w_fl, 'fimath', o_w_fimath);
    biases2_fi  = fi(biases2_float,  true, o_b_wl, o_b_fl, 'fimath', o_b_fimath);
else
    fprintf('Fixed-Point Designer not available or disabled -> running SAFE float-then-quantize mode.\n');
    % We'll keep float weights and quantize intermediate results to emulate fixed-point
    % but use double arithmetic for stable behavior.
end

%% --- Simulation Loop ---
count = 0;
total = 0;

for iter_idx = 1:iterations
    X = X_test(iter_idx, :);    % 1 x 784 (binary)
    Y = Y_test(iter_idx, :);    % 1 x 10 (one-hot)

    %% Hidden layer
    if use_fixed_point
        hidden_out_pre_relu = fi(zeros(1,10), true, h_wl, h_fl, 'fimath', h_fimath);
        % For binary inputs: sum of weights where X==1 plus bias
        for neuron = 1:10
            curr_w = weights1_fi(:, neuron)';         % 1 x 784 fi
            % logical index must be logical
            active_w = curr_w(logical(X));           % select only weights where input=1
            % sum with fi arithmetic (saturate/round as defined)
            neuron_sum = sum(active_w);               % fi sum
            neuron_sum = neuron_sum + biases1_fi(neuron);
            hidden_out_pre_relu(neuron) = neuron_sum;
        end
        % ReLU (with fixed-point): negative -> 0
        hidden_out_double = double(hidden_out_pre_relu);
        hidden_out_double(hidden_out_double < 0) = 0;
        hidden_out = fi(hidden_out_double, true, h_wl, h_fl, 'fimath', h_fimath);
    else
        % float-then-quantize mode
        hidden_out_pre_relu = zeros(1,10);
        for neuron = 1:10
            curr_w = weights1_float(:, neuron)';      % 1 x 784
            neuron_sum = sum(curr_w(logical(X)));     % float sum
            neuron_sum = neuron_sum + biases1_float(neuron);
            hidden_out_pre_relu(neuron) = neuron_sum;
        end
        hidden_out_pre_relu(hidden_out_pre_relu < 0) = 0;
        % Quantize to Q5.2 to emulate hidden fixed-point
        scale_h = 2^h_fl;
        hidden_out_q = round(hidden_out_pre_relu * scale_h) / scale_h;
        hidden_out = hidden_out_q; % double array but quantized values
    end

    %% Output layer
    if use_fixed_point
        output_out_fi = fi(zeros(1,10), true, o_b_wl, o_b_fl, 'fimath', o_b_fimath);
        for neuron = 1:10
            curr_w = weights2_fi(:, neuron)';   % 1 x 10 fi (Q2.13)
            curr_b = biases2_fi(neuron);         % fi Q13.2

            neuron_sum = fi(0, true, o_b_wl, o_b_fl, 'fimath', o_b_fimath);

            % Multiply hidden_out (Q5.2) by curr_w (Q2.13)
            % product_full is handled by fimath rules; then re-quantize to Q13.2
            for i = 1:10
                product_full = hidden_out(i) * curr_w(i); % fi times fi -> fi with combined fimath
                product_re_quant = fi(product_full, true, o_b_wl, o_b_fl, 'fimath', o_b_fimath);
                neuron_sum = neuron_sum + product_re_quant;
            end

            neuron_sum = neuron_sum + curr_b;
            output_out_fi(neuron) = neuron_sum;
        end

        % For decision, convert to double for argmax
        output_vals = double(output_out_fi);
    else
        % float mode: compute in double then quantize to Q13.2
        neuron_sums = zeros(1,10);
        for neuron = 1:10
            curr_w = weights2_float(:, neuron)';   % 1 x 10
            curr_b = biases2_float(neuron);
            prod = hidden_out .* curr_w;           % elementwise
            s = sum(prod) + curr_b;                % float
            neuron_sums(neuron) = s;
        end
        % Quantize to Q13.2
        scale_o_b = 2^o_b_fl;
        output_vals = round(neuron_sums * scale_o_b) / scale_o_b;
    end

    %% Prediction and accuracy
    [~, prediction_idx] = max(output_vals);
    prediction = zeros(1,10);
    prediction(prediction_idx) = 1;

    if isequal(prediction, Y)
        count = count + 1;
    end
    total = total + 1;

    if mod(total, 100) == 0
        fprintf('Processed %d samples.\n', total);
    end
end

fprintf('\nAccuracy: %d / %d = %.2f%%\n', count, total, (count/total)*100);
