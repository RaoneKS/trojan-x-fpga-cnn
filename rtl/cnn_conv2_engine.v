`timescale 1ns/1ps

module cnn_conv2_engine (
    input wire clk,
    input wire rst,
    input wire start,

    output reg done,
    output reg [11:0] output_count
);

    // Input: 14x14x8 = 1568
    reg signed [31:0] pool1 [0:1567];

    // Conv2 weights: 16 filters x 8 channels x 3x3
    reg signed [7:0] weight [0:1151];

    // Conv2 biases
    reg signed [7:0] bias [0:15];

    // Output: 14x14x16 = 3136
    reg signed [31:0] conv2 [0:3135];

    localparam IDLE = 3'd0;
    localparam MAC  = 3'd1;
    localparam SAVE = 3'd2;
    localparam FINISH = 3'd3;

    reg [2:0] state;

    integer row;
    integer col;
    integer out_ch;
    integer in_ch;
    integer kr;
    integer kc;

    integer in_r;
    integer in_c;
    integer input_addr;
    integer weight_addr;
    integer output_addr;

    reg signed [63:0] accumulator;

    initial begin
        $readmemh("data/pool1_output.mem", pool1);
        $readmemh("data/int8/conv2_weight.mem", weight);
        $readmemh("data/int8/conv2_bias.mem", bias);

        state = IDLE;
        done = 1'b0;
        output_count = 0;

        row = 0;
        col = 0;
        out_ch = 0;
        in_ch = 0;
        kr = 0;
        kc = 0;
        accumulator = 0;
    end

    always @(posedge clk) begin

        if (rst) begin
            state <= IDLE;
            done <= 1'b0;
            output_count <= 0;

            row <= 0;
            col <= 0;
            out_ch <= 0;
            in_ch <= 0;
            kr <= 0;
            kc <= 0;

            accumulator <= 0;
        end

        else begin

            case (state)

                IDLE: begin
                    done <= 1'b0;

                    if (start) begin
                        row <= 0;
                        col <= 0;
                        out_ch <= 0;
                        in_ch <= 0;
                        kr <= 0;
                        kc <= 0;

                        accumulator <= bias[0];

                        state <= MAC;
                    end
                end

                MAC: begin

                    in_r = row + kr - 1;
                    in_c = col + kc - 1;

                    if ((in_r >= 0) && (in_r < 14) &&
                        (in_c >= 0) && (in_c < 14)) begin

                        input_addr =
                            in_ch * 196 +
                            in_r * 14 +
                            in_c;

                        weight_addr =
                            out_ch * 72 +
                            in_ch * 9 +
                            kr * 3 +
                            kc;

                        accumulator <= accumulator +
                            pool1[input_addr] *
                            weight[weight_addr];
                    end

                    if (kc == 2) begin
                        kc <= 0;

                        if (kr == 2) begin
                            kr <= 0;

                            if (in_ch == 7) begin
                                in_ch <= 0;
                                state <= SAVE;
                            end
                            else begin
                                in_ch <= in_ch + 1;
                            end

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

                    output_addr =
                        out_ch * 196 +
                        row * 14 +
                        col;

                    // ReLU
                    if (accumulator < 0)
                        conv2[output_addr] <= 0;
                    else
                        conv2[output_addr] <= accumulator[31:0];

                    output_count <= output_count + 1'b1;

                    accumulator <= 0;

                    if (out_ch == 15) begin
                        out_ch <= 0;

                        if (col == 13) begin
                            col <= 0;

                            if (row == 13) begin
                                state <= FINISH;
                            end
                            else begin
                                row <= row + 1;
                                accumulator <= bias[0];
                                state <= MAC;
                            end

                        end
                        else begin
                            col <= col + 1;
                            accumulator <= bias[0];
                            state <= MAC;
                        end

                    end
                    else begin
                        out_ch <= out_ch + 1;
                        accumulator <= bias[out_ch + 1];
                        state <= MAC;
                    end

                end

                FINISH: begin
                    done <= 1'b1;
                end

                default: begin
                    state <= IDLE;
                end

            endcase

        end
    end

endmodule
