function main()
    model_name = "mnist_model.mat";
    % train_new_model(model_name);   % Uncomment kalau mau train
    forward(model_name, 10000);      % Jalankan forward pass
end


%% ============================================================
%  Clean & Preprocess Dataset
% ============================================================
function [X_train, Y_train, X_test, Y_test] = clean_data()
    % Load MNIST bawaan MATLAB
    digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
                                'nndatasets','DigitDataset');

    imds = imageDatastore(digitDatasetPath, ...
            'IncludeSubfolders', true, 'LabelSource', 'foldernames');

    % Split dataset (60k train, 10k test)
    [imdsTrain, imdsTest] = splitEachLabel(imds, 60000, 'randomized');

    % Convert ke array
    X_train = zeros(60000, 784);
    Y_train = zeros(60000, 10);

    X_test = zeros(10000, 784);
    Y_test = zeros(10000, 10);

    % ------- TRAIN DATA -------
    for i = 1:60000
        img = readimage(imdsTrain, i);
        img = im2double(img);
        img = reshape(img, 1, []);
        img = img >= 0.5;                       % threshold => binary
        X_train(i,:) = img;

        lbl = imdsTrain.Labels(i);
        Y_train(i,:) = onehot(lbl);
    end

    % ------- TEST DATA -------
    for i = 1:10000
        img = readimage(imdsTest, i);
        img = im2double(img);
        img = reshape(img, 1, []);
        img = img >= 0.5;
        X_test(i,:) = img;

        lbl = imdsTest.Labels(i);
        Y_test(i,:) = onehot(lbl);
    end
end


%% ============================================================
%  Train Model
% ============================================================
function train_new_model(model_name)
    [X_train, Y_train, X_test, Y_test] = clean_data();

    layers = [
        featureInputLayer(784)
        fullyConnectedLayer(10)
        reluLayer
        fullyConnectedLayer(10)
        softmaxLayer
        classificationLayer];

    options = trainingOptions('adam', ...
        'MaxEpochs', 15, ...
        'MiniBatchSize', 128, ...
        'ValidationFrequency', 100, ...
        'Verbose', false, ...
        'Plots', 'training-progress');

    Y_train_class = vec2ind(Y_train')';
    Y_test_class = vec2ind(Y_test')';

    net = trainNetwork(X_train, categorical(Y_train_class), layers, options);

    save(model_name, "net");

    YP = classify(net, X_test);
    acc = sum(YP == categorical(Y_test_class)) / numel(YP) * 100;

    fprintf("Test Accuracy: %.2f%%\n", acc);
end


%% ============================================================
%  Overflow Checking
% ============================================================
function check_overflow(x, num_bits)
    if x > (2^(num_bits-1)-1) || x < -(2^(num_bits-1))
        fprintf("Overflow detected: %d bits value exceeded.\n", num_bits);
    end
end


%% ============================================================
%  Forward Pass Fixed-Point (Mirrors Python Exact Logic)
% ============================================================
function forward(model_name, iterations)
    load(model_name, "net");

    [X_train, Y_train, X_test, Y_test] = clean_data();

    % extract weights
    W1 = net.Layers(2).Weights;
    b1 = net.Layers(2).Bias;

    W2 = net.Layers(4).Weights;
    b2 = net.Layers(4).Bias;

    % Convert to fixed 8-bit
    W1 = to_fixed(W1);  b1 = to_fixed(b1);
    W2 = to_fixed(W2);  b2 = to_fixed(b2);

    count = 0;
    total = 0;

    for idx = 1:iterations
        X = int8(X_test(idx, :));
        Y = Y_test(idx, :);

        %% -------- Hidden Layer --------
        hidden = zeros(1,10,'int8');
        for n = 1:10
            acc = int8(0);
            for k = 1:784
                if X(k) == 1
                    acc = acc + int8(W1(n,k));
                    check_overflow(acc, 8);
                end
            end
            acc = acc + int8(b1(n));
            check_overflow(acc, 8);
            hidden(n) = acc;
        end

        %% ReLU
        hidden = int8(max(hidden,0));

        %% -------- Output Layer --------
        out = zeros(1,10,'int16');
        for n = 1:10
            acc = int16(0);
            for k = 1:10
                acc = acc + int16(W2(n,k)) * int16(hidden(k));
                check_overflow(acc, 16);
            end
            acc = acc + int16(b2(n));
            out(n) = acc;
        end

        [~, pred] = max(out);
        [~, target] = max(Y);

        if pred == target
            count = count + 1;
        end

        total = total + 1;

        if mod(total,100)==0
            fprintf("%d\n", total);
        end
    end

    fprintf("Accuracy: %d / %d = %.2f%%\n", count, total, (count/total)*100);
end


%% ============================================================
%  Helper: Convert float to fixed (2 fractional bits)
% ============================================================
function fx = to_fixed(val)
    scale = 2^2; % bits_past_radix = 2
    fx = int8(round(val * scale));
end


%% ============================================================
%  One-hot Encoding Helper
% ============================================================
function y = onehot(label)
    y = zeros(1,10);
    digit = str2double(char(label));
    y(digit+1) = 1;
end
