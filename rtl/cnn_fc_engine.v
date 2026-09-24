`timescale 1ns/1ps

module cnn_fc_engine (
    input wire clk,
    input wire rst,
    input wire start,

    output reg done,
    output reg [3:0] predicted_class,
    output reg [3:0] class_count
);

    reg signed [31:0] pool2 [0:783];
    reg signed [7:0]  weight [0:7839];
    reg signed [7:0]  bias [0:9];

    reg signed [63:0] fc_out [0:9];

    localparam IDLE   = 3'd0;
    localparam MAC    = 3'd1;
    localparam SAVE   = 3'd2;
    localparam ARG    = 3'd3;
    localparam FINISH = 3'd4;

    reg [2:0] state;

    integer feature;
    integer class_idx;
    integer weight_addr;

    reg signed [63:0] accumulator;
    reg signed [63:0] product;

    reg signed [63:0] best_value;
    reg [3:0] best_class;

    initial begin
        $readmemh("data/pool2_output.mem", pool2);
        $readmemh("data/int8/fc1_weight.mem", weight);
        $readmemh("data/int8/fc1_bias.mem", bias);

        state = IDLE;
        done = 1'b0;
        predicted_class = 4'd0;
        class_count = 4'd0;

        feature = 0;
        class_idx = 0;

        accumulator = 64'sd0;
        product = 64'sd0;

        best_value = -64'sh7fffffffffffffff;
        best_class = 4'd0;
    end

    always @(posedge clk) begin

        if (rst) begin
            state <= IDLE;
            done <= 1'b0;
            predicted_class <= 4'd0;
            class_count <= 4'd0;

            feature <= 0;
            class_idx <= 0;

            accumulator <= 64'sd0;
            product <= 64'sd0;

            best_value <= -64'sh7fffffffffffffff;
            best_class <= 4'd0;
        end

        else begin

            case (state)

                IDLE: begin

                    done <= 1'b0;

                    if (start) begin

                        class_idx <= 0;
                        feature <= 0;

                        accumulator <= {{56{bias[0][7]}},bias[0]};

                        state <= MAC;
                    end

                end


                MAC: begin

                    weight_addr = class_idx * 784 + feature;

                    product =
                        $signed(pool2[feature]) *
                        $signed(weight[weight_addr]);

                    accumulator <= accumulator + product;

                    if (feature == 783) begin
                        feature <= 0;
                        state <= SAVE;
                    end
                    else begin
                        feature <= feature + 1;
                    end

                end


                SAVE: begin

                    fc_out[class_idx] <= accumulator;

                    class_count <= class_idx + 1;

                    if (class_idx == 9) begin

                        /*
                         * Class 9 is now stored.
                         * Start argmax from class 0.
                         */
                        class_idx <= 0;
                        best_value <= fc_out[0];
                        best_class <= 4'd0;

                        state <= ARG;

                    end
                    else begin

                        class_idx <= class_idx + 1;

                        accumulator <=
                            {{56{bias[class_idx + 1][7]}},
                             bias[class_idx + 1]};

                        state <= MAC;

                    end

                end


                ARG: begin

                    /*
                     * Scan classes 0..9.
                     */
                    if (fc_out[class_idx] > best_value) begin
                        best_value <= fc_out[class_idx];
                        best_class <= class_idx[3:0];
                    end

                    if (class_idx == 9) begin

                        /*
                         * Explicitly handle class 9 so that
                         * a last-element maximum is not lost
                         * because of nonblocking assignment.
                         */
                        if (fc_out[9] > best_value)
                            predicted_class <= 4'd9;
                        else
                            predicted_class <= best_class;

                        state <= FINISH;

                    end
                    else begin
                        class_idx <= class_idx + 1;
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
