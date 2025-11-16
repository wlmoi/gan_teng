// fixed_point_adder.v
// Fixed-Point Adder Module
// Adds two signed fixed-point numbers.
// Assumes both inputs have the same word length (WL) and fractional length (FL).
// The output will have WL+1 bits to accommodate potential carry-out, preventing overflow
// if the sum exceeds the original WL. For saturation/truncation, a wrapper is needed.

module fixed_point_adder #(
    parameter WL = 8,  // Word Length (e.g., 8 for Q5.2)
    parameter FL = 2   // Fractional Length (e.g., 2 for Q5.2)
) (
    input signed [WL-1:0] a,    // First operand
    input signed [WL-1:0] b,    // Second operand
    output signed [WL:0]  sum   // Sum (WL+1 bits to prevent overflow in the addition itself)
);

    // Simple addition of signed numbers.
    // Verilog handles two's complement arithmetic directly.
    assign sum = a + b;

endmodule