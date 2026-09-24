`timescale 1ns/1ps

module cnn_small_core (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    output reg         done,
    output reg [3:0]   predicted_class,
    output reg [63:0]  cycle_count
);

    /* ================================================================
       STATES
       ================================================================ */

    localparam S_IDLE  = 4'd0;
    localparam S_C1    = 4'd1;
    localparam S_C1S   = 4'd2;
    localparam S_P1    = 4'd3;
    localparam S_C2    = 4'd4;
    localparam S_C2S   = 4'd5;
    localparam S_P2    = 4'd6;
    localparam S_FC    = 4'd7;
    localparam S_ARG   = 4'd8;
    localparam S_DONE  = 4'd9;

    reg [3:0] state;

    /* ================================================================
       INPUT
       ================================================================ */

    reg signed [7:0] image [0:783];

    /* ================================================================
       WEIGHTS / BIASES
       ================================================================ */

    reg signed [7:0] c1_w [0:71];
    reg signed [7:0] c1_b [0:7];

    reg signed [7:0] c2_w [0:1151];
    reg signed [7:0] c2_b [0:15];

    reg signed [7:0] fc_w [0:7839];
    reg signed [7:0] fc_b [0:9];

    /* ================================================================
       FEATURE MAP STORAGE
       ================================================================ */

    reg signed [31:0] conv1 [0:6271];
    reg signed [31:0] pool1 [0:1567];

    reg signed [31:0] conv2 [0:3135];
    reg signed [31:0] pool2 [0:783];

    reg signed [63:0] fc_out [0:9];

    /* ================================================================
       CONVOLUTION INDICES
       ================================================================ */

    integer out_r;
    integer out_c;
    integer out_f;

    integer in_ch;

    integer kr;
    integer kc;

    integer in_r;
    integer in_c;

    integer addr;
    integer waddr;

    /* ================================================================
       POOLING INDICES
       ================================================================ */

    integer pool_r;
    integer pool_c;
    integer pool_ch;

    integer paddr;

    /* ================================================================
       FC
       ================================================================ */

    integer fc_feature;
    integer class_idx;

    /* ================================================================
       ARITHMETIC
       ================================================================ */

    reg signed [63:0] accumulator;
    reg signed [63:0] product;

    reg signed [31:0] max_value;
    reg signed [31:0] temp_value;

    reg signed [63:0] best_value;
    reg [3:0] best_class;

    /* ================================================================
       MEMORY INITIALIZATION
       ================================================================ */

    initial begin

        $readmemh("data/mnist_image0_int8.mem", image);

        $readmemh("data/int8/conv1_weight.mem", c1_w);
        $readmemh("data/int8/conv1_bias.mem", c1_b);

        $readmemh("data/int8/conv2_weight.mem", c2_w);
        $readmemh("data/int8/conv2_bias.mem", c2_b);

        $readmemh("data/int8/fc1_weight.mem", fc_w);
        $readmemh("data/int8/fc1_bias.mem", fc_b);

        state = S_IDLE;

        done = 1'b0;
        predicted_class = 4'd0;
        cycle_count = 64'd0;

        accumulator = 64'sd0;
        product = 64'sd0;

        best_value = -64'sd9223372036854775807;
        best_class = 4'd0;
    end

    /* ================================================================
       MAIN SEQUENTIAL CNN
       ================================================================ */

    always @(posedge clk) begin

        if (rst) begin

            state <= S_IDLE;

            done <= 1'b0;
            predicted_class <= 4'd0;
            cycle_count <= 64'd0;

            out_r <= 0;
            out_c <= 0;
            out_f <= 0;

            in_ch <= 0;

            kr <= 0;
            kc <= 0;

            pool_r <= 0;
            pool_c <= 0;
            pool_ch <= 0;

            fc_feature <= 0;
            class_idx <= 0;

            accumulator <= 64'sd0;
            product <= 64'sd0;

            best_value <= -64'sd9223372036854775807;
            best_class <= 4'd0;
        end

        else begin

            cycle_count <= cycle_count + 1'b1;

            case (state)

                /* ====================================================
                   IDLE
                   ==================================================== */

                S_IDLE: begin

                    done <= 1'b0;

                    if (start) begin

                        out_r <= 0;
                        out_c <= 0;
                        out_f <= 0;

                        kr <= 0;
                        kc <= 0;

                        accumulator <=
                            {{56{c1_b[0][7]}}, c1_b[0]};

                        state <= S_C1;

                    end

                end


                /* ====================================================
                   CONV1
                   ==================================================== */

                S_C1: begin

                    in_r = out_r + kr - 1;
                    in_c = out_c + kc - 1;

                    if ((in_r >= 0) && (in_r < 28) &&
                        (in_c >= 0) && (in_c < 28)) begin

                        addr =
                            in_r * 28 +
                            in_c;

                        waddr =
                            out_f * 9 +
                            kr * 3 +
                            kc;

                        product =
                            $signed(image[addr]) *
                            $signed(c1_w[waddr]);

                        accumulator <=
                            accumulator + product;

                    end

                    if (kc == 2) begin

                        kc <= 0;

                        if (kr == 2) begin

                            kr <= 0;

                            state <= S_C1S;

                        end
                        else begin

                            kr <= kr + 1;

                        end

                    end
                    else begin

                        kc <= kc + 1;

                    end

                end


                /* ====================================================
                   CONV1 SAVE + RELU
                   ==================================================== */

                S_C1S: begin

                    addr =
                        out_f * 784 +
                        out_r * 28 +
                        out_c;

                    if (accumulator < 0)
                        conv1[addr] <= 32'sd0;
                    else
                        conv1[addr] <= accumulator[31:0];

                    accumulator <= 64'sd0;

                    /* next filter */

                    if (out_f == 7) begin

                        out_f <= 0;

                        if (out_c == 27) begin

                            out_c <= 0;

                            if (out_r == 27) begin

                                out_r <= 0;
                                out_c <= 0;
                                out_f <= 0;

                                pool_r <= 0;
                                pool_c <= 0;
                                pool_ch <= 0;

                                state <= S_P1;

                            end
                            else begin

                                out_r <= out_r + 1;

                                accumulator <=
                                    {{56{c1_b[0][7]}}, c1_b[0]};

                                state <= S_C1;

                            end

                        end
                        else begin

                            out_c <= out_c + 1;

                            accumulator <=
                                {{56{c1_b[0][7]}}, c1_b[0]};

                            state <= S_C1;

                        end

                    end
                    else begin

                        out_f <= out_f + 1;

                        accumulator <=
                            {{56{c1_b[out_f+1][7]}},
                             c1_b[out_f+1]};

                        state <= S_C1;

                    end

                end


                /* ====================================================
                   MAXPOOL1
                   28x28x8 -> 14x14x8
                   ==================================================== */

                S_P1: begin

                    paddr =
                        pool_ch * 784 +
                        (pool_r * 2) * 28 +
                        (pool_c * 2);

                    max_value = conv1[paddr];

                    temp_value =
                        conv1[paddr + 1];

                    if (temp_value > max_value)
                        max_value = temp_value;

                    temp_value =
                        conv1[paddr + 28];

                    if (temp_value > max_value)
                        max_value = temp_value;

                    temp_value =
                        conv1[paddr + 29];

                    if (temp_value > max_value)
                        max_value = temp_value;

                    pool1[
                        pool_ch * 196 +
                        pool_r * 14 +
                        pool_c
                    ] <= max_value;

                    if (pool_c == 13) begin

                        pool_c <= 0;

                        if (pool_r == 13) begin

                            pool_r <= 0;

                            if (pool_ch == 7) begin

                                pool_ch <= 0;

                                out_r <= 0;
                                out_c <= 0;
                                out_f <= 0;
                                in_ch <= 0;

                                kr <= 0;
                                kc <= 0;

                                accumulator <=
                                    {{56{c2_b[0][7]}}, c2_b[0]};

                                state <= S_C2;

                            end
                            else begin

                                pool_ch <= pool_ch + 1;

                            end

                        end
                        else begin

                            pool_r <= pool_r + 1;

                        end

                    end
                    else begin

                        pool_c <= pool_c + 1;

                    end

                end


                /* ====================================================
                   CONV2
                   14x14x8 -> 14x14x16
                   ==================================================== */

                S_C2: begin

                    in_r = out_r + kr - 1;
                    in_c = out_c + kc - 1;

                    if ((in_r >= 0) && (in_r < 14) &&
                        (in_c >= 0) && (in_c < 14)) begin

                        addr =
                            in_ch * 196 +
                            in_r * 14 +
                            in_c;

                        waddr =
                            out_f * 72 +
                            in_ch * 9 +
                            kr * 3 +
                            kc;

                        product =
                            $signed(pool1[addr]) *
                            $signed(c2_w[waddr]);

                        accumulator <=
                            accumulator + product;

                    end

                    if (kc == 2) begin

                        kc <= 0;

                        if (kr == 2) begin

                            kr <= 0;

                            if (in_ch == 7) begin

                                in_ch <= 0;

                                state <= S_C2S;

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


                /* ====================================================
                   CONV2 SAVE + RELU
                   ==================================================== */

                S_C2S: begin

                    addr =
                        out_f * 196 +
                        out_r * 14 +
                        out_c;

                    if (accumulator < 0)
                        conv2[addr] <= 32'sd0;
                    else
                        conv2[addr] <= accumulator[31:0];

                    accumulator <= 64'sd0;

                    if (out_f == 15) begin

                        out_f <= 0;

                        if (out_c == 13) begin

                            out_c <= 0;

                            if (out_r == 13) begin

                                out_r <= 0;
                                out_c <= 0;
                                out_f <= 0;

                                pool_r <= 0;
                                pool_c <= 0;
                                pool_ch <= 0;

                                state <= S_P2;

                            end
                            else begin

                                out_r <= out_r + 1;

                                accumulator <=
                                    {{56{c2_b[0][7]}}, c2_b[0]};

                                state <= S_C2;

                            end

                        end
                        else begin

                            out_c <= out_c + 1;

                            accumulator <=
                                {{56{c2_b[0][7]}}, c2_b[0]};

                            state <= S_C2;

                        end

                    end
                    else begin

                        out_f <= out_f + 1;

                        accumulator <=
                            {{56{c2_b[out_f+1][7]}},
                             c2_b[out_f+1]};

                        state <= S_C2;

                    end

                end


                /* ====================================================
                   MAXPOOL2
                   14x14x16 -> 7x7x16
                   ==================================================== */

                S_P2: begin

                    paddr =
                        pool_ch * 196 +
                        (pool_r * 2) * 14 +
                        (pool_c * 2);

                    max_value = conv2[paddr];

                    temp_value =
                        conv2[paddr + 1];

                    if (temp_value > max_value)
                        max_value = temp_value;

                    temp_value =
                        conv2[paddr + 14];

                    if (temp_value > max_value)
                        max_value = temp_value;

                    temp_value =
                        conv2[paddr + 15];

                    if (temp_value > max_value)
                        max_value = temp_value;

                    pool2[
                        pool_ch * 49 +
                        pool_r * 7 +
                        pool_c
                    ] <= max_value;

                    if (pool_c == 6) begin

                        pool_c <= 0;

                        if (pool_r == 6) begin

                            pool_r <= 0;

                            if (pool_ch == 15) begin

                                pool_ch <= 0;

                                class_idx <= 0;
                                fc_feature <= 0;

                                accumulator <=
                                    {{56{fc_b[0][7]}}, fc_b[0]};

                                state <= S_FC;

                            end
                            else begin

                                pool_ch <= pool_ch + 1;

                            end

                        end
                        else begin

                            pool_r <= pool_r + 1;

                        end

                    end
                    else begin

                        pool_c <= pool_c + 1;

                    end

                end


                /* ====================================================
                   FULLY CONNECTED
                   784 -> 10
                   ==================================================== */

                S_FC: begin

                    product =
                        $signed(pool2[fc_feature]) *
                        $signed(
                            fc_w[
                                class_idx * 784 +
                                fc_feature
                            ]
                        );

                    accumulator <=
                        accumulator + product;

                    if (fc_feature == 783) begin

                        fc_out[class_idx] <=
                            accumulator + product;

                        fc_feature <= 0;

                        if (class_idx == 9) begin

                            class_idx <= 0;

                            best_value <=
                                accumulator + product;

                            best_class <= 0;

                            state <= S_ARG;

                        end
                        else begin

                            class_idx <= class_idx + 1;

                            accumulator <=
                                {{56{fc_b[class_idx+1][7]}},
                                 fc_b[class_idx+1]};

                        end

                    end
                    else begin

                        fc_feature <= fc_feature + 1;

                    end

                end


                /* ====================================================
                   ARGMAX
                   ==================================================== */

                S_ARG: begin

                    if (class_idx == 0) begin

                        best_value <= fc_out[0];
                        best_class <= 0;

                        class_idx <= 1;

                    end
                    else begin

                        if (fc_out[class_idx] > best_value) begin

                            best_value <= fc_out[class_idx];
                            best_class <= class_idx[3:0];

                        end

                        if (class_idx == 9) begin

                            if (fc_out[class_idx] > best_value)
                                predicted_class <=
                                    class_idx[3:0];
                            else
                                predicted_class <=
                                    best_class;

                            state <= S_DONE;

                        end
                        else begin

                            class_idx <= class_idx + 1;

                        end

                    end

                end


                /* ====================================================
                   DONE
                   ==================================================== */

                S_DONE: begin

                    done <= 1'b1;

                end


                default: begin

                    state <= S_IDLE;

                end

            endcase

        end

    end

endmodule
