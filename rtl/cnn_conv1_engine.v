`timescale 1ns/1ps

module cnn_conv1_engine (
    input  wire clk,
    input  wire rst,
    input  wire start,

    output reg done,
    output reg [12:0] output_count
);

    // 28x28 input image
    reg signed [7:0] image [0:783];

    // 8 filters x 3x3
    reg signed [7:0] weight [0:71];

    // 8 biases
    reg signed [7:0] bias [0:7];

    // Conv1 output: 28x28x8
    reg signed [31:0] conv1 [0:6271];

    reg [3:0] state;

    localparam IDLE = 4'd0;
    localparam MAC  = 4'd1;
    localparam SAVE = 4'd2;
    localparam DONE = 4'd3;

    integer out_r;
    integer out_c;
    integer out_f;
    integer kr;
    integer kc;

    reg signed [31:0] accumulator;

    integer in_r;
    integer in_c;
    integer image_addr;
    integer weight_addr;
    integer output_addr;

    initial begin
        $readmemh("data/mnist_image0_int8.mem", image);
        $readmemh("data/int8/conv1_weight.mem", weight);
        $readmemh("data/int8/conv1_bias.mem", bias);

        state = IDLE;
        done = 1'b0;
        output_count = 13'd0;

        out_r = 0;
        out_c = 0;
        out_f = 0;
        kr = 0;
        kc = 0;
        accumulator = 0;
    end

    always @(posedge clk) begin

        if (rst) begin
            state <= IDLE;
            done <= 1'b0;
            output_count <= 13'd0;

            out_r <= 0;
            out_c <= 0;
            out_f <= 0;
            kr <= 0;
            kc <= 0;

            accumulator <= 0;
        end
        else begin

            case (state)

                IDLE: begin
                    done <= 1'b0;

                    if (start) begin
                        out_r <= 0;
                        out_c <= 0;
                        out_f <= 0;
                        kr <= 0;
                        kc <= 0;

                        accumulator <= bias[0];

                        state <= MAC;
                    end
                end

                MAC: begin

                    in_r = out_r + kr - 1;
                    in_c = out_c + kc - 1;

                    if ((in_r >= 0) && (in_r < 28) &&
                        (in_c >= 0) && (in_c < 28)) begin

                        image_addr = in_r * 28 + in_c;
                        weight_addr = out_f * 9 + kr * 3 + kc;

                        accumulator <= accumulator +
                            image[image_addr] * weight[weight_addr];

                    end

                    if (kc == 2) begin
                        kc <= 0;

                        if (kr == 2) begin
                            kr <= 0;
                            state <= SAVE;
                        end
                        else begin
                            kr <= kr + 1;
                        end

                    end
                    else begin
                        kc <= kc + 1;
                    end

                end

                SAVE: begin

                    // ReLU
                    if (accumulator < 0)
                        conv1[out_f * 784 + out_r * 28 + out_c] <= 0;
                    else
                        conv1[out_f * 784 + out_r * 28 + out_c] <= accumulator;

                    output_count <= output_count + 1'b1;

                    accumulator <= 0;

                    if (out_f == 7) begin
                        out_f <= 0;

                        if (out_c == 27) begin
                            out_c <= 0;

                            if (out_r == 27) begin
                                state <= DONE;
                            end
                            else begin
                                out_r <= out_r + 1;
                                accumulator <= bias[0];
                                state <= MAC;
                            end
                        end
                        else begin
                            out_c <= out_c + 1;
                            accumulator <= bias[0];
                            state <= MAC;
                        end

                    end
                    else begin
                        out_f <= out_f + 1;
                        accumulator <= bias[out_f + 1];
                        state <= MAC;
                    end

                end

                DONE: begin
                    done <= 1'b1;
                end

                default: begin
                    state <= IDLE;
                end

            endcase
        end
    end

endmodule
