mport numpy as np
import keras
from keras import layers
import scipy.io # To save .mat files from Python

def clean_data(data):
    (X_train, Y_train), (X_test, Y_test) = data

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train = np.where(X_train >= 0.5, 1, 0).astype('int8')  # (60000 x 784) binary values
    X_test = np.where(X_test >= 0.5, 1, 0).astype('int8')  # (10000 x 784) binary values

    Y_train = keras.utils.to_categorical(Y_train, 10)  # (60000 x 10) 1-hot encoded
    Y_test = keras.utils.to_categorical(Y_test, 10)  # (10000 x 10) 1-hot encoded

    return X_train, Y_train, X_test, Y_test

def train_new_model(model_name):
    X_train, Y_train, X_test, Y_test = clean_data(keras.datasets.mnist.load_data())

    model = keras.Sequential(
        [
            keras.Input(shape=(784,)),
            layers.Dense(10), # Hidden layer
            layers.Activation('relu'),
            layers.Dense(10), # Output layer
            layers.Activation('softmax')
        ]
    )

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, Y_train, batch_size=128, epochs=15, validation_split=0.1)
    model.save(model_name)

    loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2)
    print("Test Loss", loss_and_metrics[0])
    print("Test Accuracy", loss_and_metrics[1])

    # --- EKSPOR BOBOT DAN BIAS KE FILE .mat ---
    weights1_float = model.layers[0].get_weights()[0]
    biases1_float = model.layers[0].get_weights()[1]
    weights2_float = model.layers[2].get_weights()[0] # Layer index 2 for the second Dense layer
    biases2_float = model.layers[2].get_weights()[1]

    # Save as .mat file for MATLAB
    scipy.io.savemat('model_weights.mat', {
        'weights1_float': weights1_float,
        'biases1_float': biases1_float,
        'weights2_float': weights2_float,
        'biases2_float': biases2_float
    })
    print("Model weights and biases exported to model_weights.mat")

    # --- EKSPOR X_test dan Y_test yang sudah dibersihkan ---
    scipy.io.savemat('mnist_test_data.mat', {
        'X_test_binary': X_test,
        'Y_test_onehot': Y_test
    })
    print("X_test and Y_test exported to mnist_test_data.mat")


def check_overflow(x, num_bits):
    # This function is not used in the main MATLAB simulation,
    # but good to keep for Python's forward pass if needed.
    pass # Placeholder, as actual check is in MATLAB's fi objects.

def forward(model_name, iterations=10000):
    X_train, Y_train, X_test, Y_test = clean_data(keras.datasets.mnist.load_data())

    model = keras.saving.load_model(model_name)
    weights1 = model.layers[0].get_weights()[0]
    biases1 = model.layers[0].get_weights()[1]
    weights2 = model.layers[2].get_weights()[0]
    biases2 = model.layers[2].get_weights()[1]

    # --- Fixed-point conversion function (match MATLAB's format) ---
    # We will use the *same* fixed-point configuration in Python's forward pass
    # as in MATLAB to ensure consistency for comparison.
    # From MATLAB: h_wl=8, h_fl=2 (Q5.2 for layer 1 params and output)
    # o_w_wl=16, o_w_fl=13 (Q2.13 for layer 2 weights)
    # o_b_wl=16, o_b_fl=2 (Q13.2 for layer 2 biases and output)

    def to_fixed_h(float_value): # For hidden layer weights/biases
        scale = 2 ** 2 # 2 fractional bits
        val = float_value * scale
        val_round = int(np.round(val)) # Use numpy round for consistency
        # MATLAB's fi handles saturation/overflow, here we just return the integer
        # For actual overflow check, you'd add:
        # if val_round > (2**(8-1)-1) or val_round < -(2**(8-1)): print("Overflow")
        return val_round

    def to_fixed_ow(float_value): # For output layer weights
        scale = 2 ** 13 # 13 fractional bits
        val = float_value * scale
        val_round = int(np.round(val))
        return val_round

    def to_fixed_ob(float_value): # For output layer biases/output (before argmax)
        scale = 2 ** 2 # 2 fractional bits
        val = float_value * scale
        val_round = int(np.round(val))
        return val_round

    # Apply fixed-point conversion
    weights1_fp = np.vectorize(to_fixed_h)(weights1).astype('int8')
    biases1_fp = np.vectorize(to_fixed_h)(biases1).astype('int8')
    weights2_fp = np.vectorize(to_fixed_ow)(weights2).astype('int16')
    biases2_fp = np.vectorize(to_fixed_ob)(biases2).astype('int16') # biases2 are Q13.2

    count = 0
    total = 0

    for X_sample, Y_sample in zip(X_test, Y_test):
        # HIDDEN LAYER
        # Manual loop for fixed-point simulation (similar to MATLAB)
        hidden_layer_output_pre_relu = np.zeros(10, dtype=np.int8)
        
        for neuron_idx in range(10):
            current_sum = 0
            # For binary input, it's sum of active weights
            for pixel_idx in range(784):
                if X_sample[pixel_idx] == 1:
                    current_sum += weights1_fp[pixel_idx, neuron_idx]
            
            # Add bias
            current_sum += biases1_fp[neuron_idx]
            
            # Saturate to 8-bit signed range (Q5.2 equivalent max/min integer value)
            max_8bit_signed = (2**(8-1))-1 # 127
            min_8bit_signed = -(2**(8-1))   # -128
            current_sum = np.clip(current_sum, min_8bit_signed, max_8bit_signed)
            
            hidden_layer_output_pre_relu[neuron_idx] = current_sum

        # RELU (applied after converting to Python int, then back to fixed-point int)
        hidden_layer_output_relu = np.maximum(0, hidden_layer_output_pre_relu).astype(np.int8)

        # OUTPUT LAYER
        output_layer_output = np.zeros(10, dtype=np.int32) # Use int32 for temporary sum before final clip

        for neuron_idx in range(10):
            current_sum_out_layer = 0
            for input_idx in range(10):
                # Multiplication: Q5.2 (hidden_layer_output_relu) * Q2.13 (weights2_fp)
                # This product is Q7.15 (total 23 bits).
                product_val = np.int32(hidden_layer_output_relu[input_idx]) * np.int32(weights2_fp[input_idx, neuron_idx])
                
                # Re-quantize product from Q7.15 to Q13.2 before summing
                # Q7.15 to Q13.2 means shift right by 15 - 2 = 13 bits.
                # Assuming simple truncation here for Verilog consistency.
                re_quant_shift = 13
                re_quant_product = product_val // (2 ** re_quant_shift) # Integer division for truncation
                
                current_sum_out_layer += re_quant_product

            # Add bias (Q13.2)
            current_sum_out_layer += biases2_fp[neuron_idx]
            
            # Saturate to 16-bit signed range (Q13.2 equivalent max/min integer value)
            max_16bit_signed = (2**(16-1))-1 # 32767
            min_16bit_signed = -(2**(16-1))   # -32768
            current_sum_out_layer = np.clip(current_sum_out_layer, min_16bit_signed, max_16bit_signed)
            
            output_layer_output[neuron_idx] = current_sum_out_layer

        # Prediction (Argmax)
        prediction_fixed = np.where(output_layer_output == np.max(output_layer_output), 1, 0).astype(np.int16)

        if np.array_equal(prediction_fixed, Y_sample):
            count += 1
        total+=1

        if total >= iterations:
            break

        if total % 100 == 0:
            print(f'Python Fixed-Point Sim: Processed {total} samples.')


    print(f'Python Fixed-Point Sim Accuracy: {count} / {total} = {count / total * 100:.2f}%')


def main():
    model_name = 'mnist_model.keras'
    train_new_model(model_name) # Run this to train the model and export data
    forward(model_name, 10000) # You can then run this to compare fixed-point in Python


if __name__ == '__main__':
    main()
