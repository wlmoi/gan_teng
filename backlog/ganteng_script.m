function main()
    model_name = "model.mat";   % file berisi W1,b1,W2,b2
    forward(model_name, 10000);
end

%% ========================================================================
function [X_train, Y_train, X_test, Y_test] = clean_data()

    digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
                                'nndatasets','DigitDataset');

    imds = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders', true, 'LabelSource', 'foldernames');

    % 70% train, 30% test
    [imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, "randomized");

    numTrain = numel(imdsTrain.Files);
    numTest  = numel(imdsTest.Files);

    X_train = zeros(numTrain, 784);
    Y_train = zeros(numTrain, 10);

    X_test = zeros(numTest, 784);
    Y_test = zeros(numTest, 10);

    % TRAIN
    for idx = 1:numTrain
        img = im2double(readimage(imdsTrain, idx));
        img = img(:)' >= 0.5;
        X_train(idx,:) = img;
        lbl = str2double(string(imdsTrain.Labels(idx)));
        Y_train(idx,:) = onehot(lbl);
    end

    % TEST
    for idx = 1:numTest
        img = im2double(readimage(imdsTest, idx));
        img = img(:)' >= 0.5;
        X_test(idx,:) = img;
        lbl = str2double(string(imdsTest.Labels(idx)));
        Y_test(idx,:) = onehot(lbl);
    end
end

%% ========================================================================
function forward(model_name, iterations)

    % Load weight manual, BUKAN net
    M = load(model_name);

    W1 = to_fixed(M.W1);
    b1 = to_fixed(M.b1(:));
    W2 = to_fixed(M.W2);
    b2 = to_fixed(M.b2(:));

    [~, ~, X_test, Y_test] = clean_data();

    correct = 0;
    actual = zeros(iterations,1);
    predict = zeros(iterations,1);

    wb = waitbar(0, 'Running Fixed-Point Inference...');

    for idx = 1:iterations

        X = int8(X_test(idx, :));
        Y = Y_test(idx, :);

        % -----------------------------
        % Hidden Layer
        % -----------------------------
        hidden = zeros(10,1,'int8');
        for n = 1:10
            acc = int16(0);
            for k = 1:784
                if X(k) == 1
                    acc = acc + int16(W1(n,k));
                end
            end
            acc = acc + int16(b1(n));
            hidden(n) = int8(max(acc,0));
        end

        % -----------------------------
        % Output Layer
        % -----------------------------
        out = zeros(10,1,'int16');
        for n = 1:10
            acc = int16(0);
            for k = 1:10
                acc = acc + int16(W2(n,k)) * int16(hidden(k));
            end
            out(n) = acc + int16(b2(n));
        end

        [~, pred] = max(out);
        [~, target] = max(Y);

        actual(idx) = target - 1;
        predict(idx) = pred - 1;

        if pred == target
            correct = correct + 1;
        end

        waitbar(idx/iterations, wb);
    end

    close(wb);

    fprintf("Accuracy: %d / %d = %.2f%%\n", ...
        correct, iterations, correct/iterations * 100);

    figure;
    confusionchart(actual, predict);
    title('Confusion Matrix (Fixed-Point Inference)');
end

%% ========================================================================
function fx = to_fixed(val)
    fx = int8(round(val * 4));   % 2 fractional bits
end

%% ========================================================================
function y = onehot(digit)
    y = zeros(1,10);
    y(digit+1) = 1;
end
