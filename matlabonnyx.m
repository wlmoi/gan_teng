model = importONNXNetwork("mnist_model.onnx", ...
                          "InputDataFormats","BC", ...
                          "OutputDataFormats","BC");
lgraph = importONNXLayers("mnist_model.onnx");

% Modify if required
lgraph = removeLayers(lgraph,"softmax_output");
lgraph = addLayers(lgraph,softmaxLayer("Name","softmax"));
lgraph = connectLayers(lgraph,"dense_1","softmax");

analyzeNetwork(lgraph)
