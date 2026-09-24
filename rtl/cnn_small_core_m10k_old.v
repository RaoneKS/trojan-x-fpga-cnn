`timescale 1ns/1ps

/*
 * RAM-based sequential MNIST CNN accelerator
 *
 * 28x28x1
 *   -> Conv1 3x3, 8 filters + ReLU
 *   -> MaxPool 2x2
 *   -> Conv2 3x3, 16 filters + ReLU
 *   -> MaxPool 2x2
 *   -> FC 784 -> 10
 *   -> Argmax
 *
 * Large arrays use synchronous single-port RAMs with ramstyle=M10K.
 * This is intended to prevent Quartus from converting the CNN into
 * hundreds of thousands of flip-flops.
 *
 * Numerical convention matches the previously verified Python/RTL
 * reference: signed INT8 weights/biases, signed MAC, direct INT8 bias,
 * ReLU after convolution.
 */


/* ================================================================
 * Generic synchronous single-port RAM
 * ================================================================ */
module cnn_sp_ram #(
    parameter WIDTH = 8,
    parameter DEPTH = 1,
    parameter AW = 1,
    parameter INIT_FILE = ""
)(
    input  wire             clk,
    input  wire             we,
    input  wire [AW-1:0]    waddr,
    input  wire [WIDTH-1:0] wdata,
    input  wire [AW-1:0]    raddr,
    output reg  [WIDTH-1:0] rdata
);

    (* ramstyle = "M10K" *)
    reg [WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        if (INIT_FILE != "")
            $readmemh(INIT_FILE, mem);
    end

    always @(posedge clk) begin
        if (we)
            mem[waddr] <= wdata;

        rdata <= mem[raddr];
    end

endmodule


/* ================================================================
 * CNN CORE
 * ================================================================ */
module cnn_small_core (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,

    output reg         done,
    output reg [3:0]   predicted_class,
    output reg [63:0]  cycle_count
);

    /* ============================================================
     * FSM states
     * ============================================================ */
    localparam S_IDLE  = 5'd0;

    localparam S_C1_ADDR = 5'd1;
    localparam S_C1_MAC  = 5'd2;
    localparam S_C1_SAVE = 5'd3;

    localparam S_P1_ADDR = 5'd4;
    localparam S_P1_R0   = 5'd5;
    localparam S_P1_R1   = 5'd6;
    localparam S_P1_R2   = 5'd7;
    localparam S_P1_R3   = 5'd8;
    localparam S_P1_SAVE = 5'd9;

    localparam S_C2_ADDR = 5'd10;
    localparam S_C2_MAC  = 5'd11;
    localparam S_C2_SAVE = 5'd12;

    localparam S_P2_ADDR = 5'd13;
    localparam S_P2_R0   = 5'd14;
    localparam S_P2_R1   = 5'd15;
    localparam S_P2_R2   = 5'd16;
    localparam S_P2_R3   = 5'd17;
    localparam S_P2_SAVE = 5'd18;

    localparam S_FC_ADDR = 5'd19;
    localparam S_FC_MAC  = 5'd20;
    localparam S_FC_SAVE = 5'd21;

    localparam S_ARG     = 5'd22;
    localparam S_DONE    = 5'd23;

    reg [4:0] state;


    /* ============================================================
     * INPUT / WEIGHT RAMS
     * ============================================================ */

    reg [9:0]  image_raddr;
    reg [6:0]  c1w_raddr;
    reg [10:0] c2w_raddr;
    reg [12:0] fcw_raddr;

    wire signed [7:0] image_q;
    wire signed [7:0] c1w_q;
    wire signed [7:0] c2w_q;
    wire signed [7:0] fcw_q;


    cnn_sp_ram #(
        .WIDTH(8),
        .DEPTH(784),
        .AW(10),
        .INIT_FILE("data/mnist_image0_int8.mem")
    ) u_image_ram (
        .clk(clk),
        .we(1'b0),
        .waddr(10'd0),
        .wdata(8'd0),
        .raddr(image_raddr),
        .rdata(image_q)
    );


    cnn_sp_ram #(
        .WIDTH(8),
        .DEPTH(72),
        .AW(7),
        .INIT_FILE("data/int8/conv1_weight.mem")
    ) u_c1_weight_ram (
        .clk(clk),
        .we(1'b0),
        .waddr(7'd0),
        .wdata(8'd0),
        .raddr(c1w_raddr),
        .rdata(c1w_q)
    );


    cnn_sp_ram #(
        .WIDTH(8),
        .DEPTH(1152),
        .AW(11),
        .INIT_FILE("data/int8/conv2_weight.mem")
    ) u_c2_weight_ram (
        .clk(clk),
        .we(1'b0),
        .waddr(11'd0),
        .wdata(8'd0),
        .raddr(c2w_raddr),
        .rdata(c2w_q)
    );


    cnn_sp_ram #(
        .WIDTH(8),
        .DEPTH(7840),
        .AW(13),
        .INIT_FILE("data/int8/fc1_weight.mem")
    ) u_fc_weight_ram (
        .clk(clk),
        .we(1'b0),
        .waddr(13'd0),
        .wdata(8'd0),
        .raddr(fcw_raddr),
        .rdata(fcw_q)
    );


    /* ============================================================
     * BIAS REGISTERS
     * ============================================================ */

    reg signed [7:0] c1_bias [0:7];
    reg signed [7:0] c2_bias [0:15];
    reg signed [7:0] fc_bias [0:9];

    initial begin
        $readmemh("data/int8/conv1_bias.mem", c1_bias);
        $readmemh("data/int8/conv2_bias.mem", c2_bias);
        $readmemh("data/int8/fc1_bias.mem", fc_bias);
    end


    /* ============================================================
     * FEATURE MAP RAMS
     *
     * Conv1  = 28*28*8  = 6272
     * Pool1  = 14*14*8  = 1568
     * Conv2  = 14*14*16 = 3136
     * Pool2  = 7*7*16   = 784
     * ============================================================ */

    reg [12:0] c1_waddr;
    reg [10:0] p1_waddr;
    reg [11:0] c2_waddr;
    reg [9:0]  p2_waddr;

    reg [12:0] c1_raddr;
    reg [10:0] p1_raddr;
    reg [11:0] c2_raddr;
    reg [9:0]  p2_raddr;

    reg signed [31:0] c1_wdata;
    reg signed [31:0] p1_wdata;
    reg signed [31:0] c2_wdata;
    reg signed [31:0] p2_wdata;

    wire signed [31:0] c1_q;
    wire signed [31:0] p1_q;
    wire signed [31:0] c2_q;
    wire signed [31:0] p2_q;


    /*
     * Write enables are generated combinationally.
     * They are HIGH only in the corresponding SAVE state.
     */
    wire c1_we = (state == S_C1_SAVE);
    wire p1_we = (state == S_P1_SAVE);
    wire c2_we = (state == S_C2_SAVE);
    wire p2_we = (state == S_P2_SAVE);


    cnn_sp_ram #(
        .WIDTH(32),
        .DEPTH(6272),
        .AW(13)
    ) u_conv1_ram (
        .clk(clk),
        .we(c1_we),
        .waddr(c1_waddr),
        .wdata(c1_wdata),
        .raddr(c1_raddr),
        .rdata(c1_q)
    );


    cnn_sp_ram #(
        .WIDTH(32),
        .DEPTH(1568),
        .AW(11)
    ) u_pool1_ram (
        .clk(clk),
        .we(p1_we),
        .waddr(p1_waddr),
        .wdata(p1_wdata),
        .raddr(p1_raddr),
        .rdata(p1_q)
    );


    cnn_sp_ram #(
        .WIDTH(32),
        .DEPTH(3136),
        .AW(12)
    ) u_conv2_ram (
        .clk(clk),
        .we(c2_we),
        .waddr(c2_waddr),
        .wdata(c2_wdata),
        .raddr(c2_raddr),
        .rdata(c2_q)
    );


    cnn_sp_ram #(
        .WIDTH(32),
        .DEPTH(784),
        .AW(10)
    ) u_pool2_ram (
        .clk(clk),
        .we(p2_we),
        .waddr(p2_waddr),
        .wdata(p2_wdata),
        .raddr(p2_raddr),
        .rdata(p2_q)
    );


    /* ============================================================
     * CNN COUNTERS
     * ============================================================ */

    integer out_r;
    integer out_c;
    integer out_f;

    integer in_ch;

    integer kr;
    integer kc;

    integer pool_r;
    integer pool_c;
    integer pool_ch;

    integer fc_feature;
    integer class_idx;


    /* Temporary address calculations */
    integer ir;
    integer ic;
    integer addr;
    integer waddr;


    /* ============================================================
     * ACCUMULATORS
     * ============================================================ */

    reg signed [63:0] accumulator;
    reg signed [63:0] product;


    /* ============================================================
     * MAXPOOL
     * ============================================================ */

    reg signed [31:0] pool_max;
    reg signed [31:0] pool_temp;


    /* ============================================================
     * ARGMAX
     * ============================================================ */

    reg signed [63:0] best_value;
    reg [3:0] best_class;


    /* ============================================================
     * COMBINATIONAL WRITE DATA / ADDRESSES
     * ============================================================ */

    always @(*) begin

        c1_waddr  = 13'd0;
        c1_wdata  = 32'd0;

        p1_waddr  = 11'd0;
        p1_wdata  = 32'd0;

        c2_waddr  = 12'd0;
        c2_wdata  = 32'd0;

        p2_waddr  = 10'd0;
        p2_wdata  = 32'd0;


        /* Conv1 output */
        if (state == S_C1_SAVE) begin

            c1_waddr = out_f * 784 +
                       out_r * 28 +
                       out_c;

            if (accumulator < 0)
                c1_wdata = 32'd0;
            else
                c1_wdata = accumulator[31:0];

        end


        /* Pool1 output */
        if (state == S_P1_SAVE) begin

            p1_waddr = pool_ch * 196 +
                       pool_r * 14 +
                       pool_c;

            p1_wdata = pool_max;

        end


        /* Conv2 output */
        if (state == S_C2_SAVE) begin

            c2_waddr = out_f * 196 +
                       out_r * 14 +
                       out_c;

            if (accumulator < 0)
                c2_wdata = 32'd0;
            else
                c2_wdata = accumulator[31:0];

        end


        /* Pool2 output */
        if (state == S_P2_SAVE) begin

            p2_waddr = pool_ch * 49 +
                       pool_r * 7 +
                       pool_c;

            p2_wdata = pool_max;

        end

    end


    /* ============================================================
     * MAIN SEQUENTIAL CONTROLLER
     * ============================================================ */

    always @(posedge clk) begin

        if (rst) begin

            state <= S_IDLE;

            done <= 1'b0;
            predicted_class <= 4'd0;
            cycle_count <= 64'd0;


            image_raddr <= 10'd0;
            c1w_raddr <= 7'd0;
            c2w_raddr <= 11'd0;
            fcw_raddr <= 13'd0;


            c1_raddr <= 13'd0;
            p1_raddr <= 11'd0;
            c2_raddr <= 12'd0;
            p2_raddr <= 10'd0;


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

            pool_max <= 32'sd0;
            pool_temp <= 32'sd0;


            best_value <= -64'sh7fffffffffffffff;
            best_class <= 4'd0;

        end

        else begin

            cycle_count <= cycle_count + 1'b1;


            case (state)


                /* =================================================
                 * IDLE
                 * ================================================= */

                S_IDLE: begin

                    done <= 1'b0;

                    if (start) begin

                        out_r <= 0;
                        out_c <= 0;
                        out_f <= 0;

                        kr <= 0;
                        kc <= 0;

                        accumulator <=
                            {{56{c1_bias[0][7]}}, c1_bias[0]};

                        state <= S_C1_ADDR;

                    end

                end


                /* =================================================
                 * CONV1 ADDRESS
                 * ================================================= */

                S_C1_ADDR: begin

                    ir = out_r + kr - 1;
                    ic = out_c + kc - 1;


                    if ((ir >= 0) &&
                        (ir < 28) &&
                        (ic >= 0) &&
                        (ic < 28)) begin

                        image_raddr <= ir * 28 + ic;

                        c1w_raddr <=
                            out_f * 9 +
                            kr * 3 +
                            kc;

                    end

                    else begin

                        image_raddr <= 10'd0;
                        c1w_raddr <= 7'd0;

                    end


                    state <= S_C1_MAC;

                end


                /* =================================================
                 * CONV1 MAC
                 * ================================================= */

                S_C1_MAC: begin

                    ir = out_r + kr - 1;
                    ic = out_c + kc - 1;


                    if ((ir >= 0) &&
                        (ir < 28) &&
                        (ic >= 0) &&
                        (ic < 28)) begin

                        product =
                            $signed(image_q) *
                            $signed(c1w_q);

                        accumulator <=
                            accumulator + product;

                    end


                    if (kc == 2) begin

                        kc <= 0;

                        if (kr == 2) begin

                            kr <= 0;

                            state <= S_C1_SAVE;

                        end

                        else begin

                            kr <= kr + 1;

                        end

                    end

                    else begin

                        kc <= kc + 1;

                    end

                end


                /* =================================================
                 * CONV1 SAVE
                 * ================================================= */

                S_C1_SAVE: begin

                    accumulator <= 64'sd0;


                    if (out_f == 7) begin

                        out_f <= 0;


                        if (out_c == 27) begin

                            out_c <= 0;


                            if (out_r == 27) begin

                                out_r <= 0;

                                pool_r <= 0;
                                pool_c <= 0;
                                pool_ch <= 0;

                                state <= S_P1_ADDR;

                            end

                            else begin

                                out_r <= out_r + 1;

                                accumulator <=
                                    {{56{c1_bias[0][7]}},
                                     c1_bias[0]};

                                state <= S_C1_ADDR;

                            end

                        end

                        else begin

                            out_c <= out_c + 1;

                            accumulator <=
                                {{56{c1_bias[0][7]}},
                                 c1_bias[0]};

                            state <= S_C1_ADDR;

                        end

                    end

                    else begin

                        out_f <= out_f + 1;

                        accumulator <=
                            {{56{c1_bias[out_f+1][7]}},
                             c1_bias[out_f+1]};

                        state <= S_C1_ADDR;

                    end

                end


                /* =================================================
                 * MAXPOOL1 ADDRESS
                 * ================================================= */

                S_P1_ADDR: begin

                    c1_raddr <=
                        pool_ch * 784 +
                        (pool_r * 2) * 28 +
                        (pool_c * 2);

                    pool_max <= -32'sh80000000;

                    state <= S_P1_R0;

                end


                /* =================================================
                 * MAXPOOL1 READ 0
                 * ================================================= */

                S_P1_R0: begin

                    pool_temp = c1_q;

                    if (pool_temp > pool_max)
                        pool_max <= pool_temp;


                    c1_raddr <=
                        pool_ch * 784 +
                        (pool_r * 2) * 28 +
                        (pool_c * 2) + 1;

                    state <= S_P1_R1;

                end


                /* =================================================
                 * MAXPOOL1 READ 1
                 * ================================================= */

                S_P1_R1: begin

                    pool_temp = c1_q;

                    if (pool_temp > pool_max)
                        pool_max <= pool_temp;


                    c1_raddr <=
                        pool_ch * 784 +
                        (pool_r * 2 + 1) * 28 +
                        (pool_c * 2);

                    state <= S_P1_R2;

                end


                /* =================================================
                 * MAXPOOL1 READ 2
                 * ================================================= */

                S_P1_R2: begin

                    pool_temp = c1_q;

                    if (pool_temp > pool_max)
                        pool_max <= pool_temp;


                    c1_raddr <=
                        pool_ch * 784 +
                        (pool_r * 2 + 1) * 28 +
                        (pool_c * 2) + 1;

                    state <= S_P1_R3;

                end


                /* =================================================
                 * MAXPOOL1 READ 3
                 * ================================================= */

                S_P1_R3: begin

                    pool_temp = c1_q;

                    if (pool_temp > pool_max)
                        pool_max <= pool_temp;


                    state <= S_P1_SAVE;

                end


                /* =================================================
                 * MAXPOOL1 SAVE
                 * ================================================= */

                S_P1_SAVE: begin

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
                                    {{56{c2_bias[0][7]}},
                                     c2_bias[0]};

                                state <= S_C2_ADDR;

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


                /* =================================================
                 * CONV2 ADDRESS
                 * ================================================= */

                S_C2_ADDR: begin

                    ir = out_r + kr - 1;
                    ic = out_c + kc - 1;


                    if ((ir >= 0) &&
                        (ir < 14) &&
                        (ic >= 0) &&
                        (ic < 14)) begin

                        p1_raddr <=
                            in_ch * 196 +
                            ir * 14 +
                            ic;


                        c2w_raddr <=
                            out_f * 72 +
                            in_ch * 9 +
                            kr * 3 +
                            kc;

                    end

                    else begin

                        p1_raddr <= 11'd0;
                        c2w_raddr <= 11'd0;

                    end


                    state <= S_C2_MAC;

                end


                /* =================================================
                 * CONV2 MAC
                 * ================================================= */

                S_C2_MAC: begin

                    ir = out_r + kr - 1;
                    ic = out_c + kc - 1;


                    if ((ir >= 0) &&
                        (ir < 14) &&
                        (ic >= 0) &&
                        (ic < 14)) begin

                        product =
                            $signed(p1_q) *
                            $signed(c2w_q);

                        accumulator <=
                            accumulator + product;

                    end


                    if (kc == 2) begin

                        kc <= 0;


                        if (kr == 2) begin

                            kr <= 0;


                            if (in_ch == 7) begin

                                in_ch <= 0;

                                state <= S_C2_SAVE;

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


                /* =================================================
                 * CONV2 SAVE
                 * ================================================= */

                S_C2_SAVE: begin

                    accumulator <= 64'sd0;


                    if (out_f == 15) begin

                        out_f <= 0;


                        if (out_c == 13) begin

                            out_c <= 0;


                            if (out_r == 13) begin

                                out_r <= 0;

                                pool_r <= 0;
                                pool_c <= 0;
                                pool_ch <= 0;

                                state <= S_P2_ADDR;

                            end

                            else begin

                                out_r <= out_r + 1;

                                accumulator <=
                                    {{56{c2_bias[0][7]}},
                                     c2_bias[0]};

                                state <= S_C2_ADDR;

                            end

                        end

                        else begin

                            out_c <= out_c + 1;

                            accumulator <=
                                {{56{c2_bias[0][7]}},
                                 c2_bias[0]};

                            state <= S_C2_ADDR;

                        end

                    end

                    else begin

                        out_f <= out_f + 1;

                        accumulator <=
                            {{56{c2_bias[out_f+1][7]}},
                             c2_bias[out_f+1]};

                        state <= S_C2_ADDR;

                    end

                end


                /* =================================================
                 * MAXPOOL2 ADDRESS
                 * ================================================= */

                S_P2_ADDR: begin

                    c2_raddr <=
                        pool_ch * 196 +
                        (pool_r * 2) * 14 +
                        (pool_c * 2);

                    pool_max <= -32'sh80000000;

                    state <= S_P2_R0;

                end


                /* =================================================
                 * MAXPOOL2 READ 0
                 * ================================================= */

                S_P2_R0: begin

                    pool_temp = c2_q;

                    if (pool_temp > pool_max)
                        pool_max <= pool_temp;


                    c2_raddr <=
                        pool_ch * 196 +
                        (pool_r * 2) * 14 +
                        (pool_c * 2) + 1;

                    state <= S_P2_R1;

                end


                /* =================================================
                 * MAXPOOL2 READ 1
                 * ================================================= */

                S_P2_R1: begin

                    pool_temp = c2_q;

                    if (pool_temp > pool_max)
                        pool_max <= pool_temp;


                    c2_raddr <=
                        pool_ch * 196 +
                        (pool_r * 2 + 1) * 14 +
                        (pool_c * 2);

                    state <= S_P2_R2;

                end


                /* =================================================
                 * MAXPOOL2 READ 2
                 * ================================================= */

                S_P2_R2: begin

                    pool_temp = c2_q;

                    if (pool_temp > pool_max)
                        pool_max <= pool_temp;


                    c2_raddr <=
                        pool_ch * 196 +
                        (pool_r * 2 + 1) * 14 +
                        (pool_c * 2) + 1;

                    state <= S_P2_R3;

                end


                /* =================================================
                 * MAXPOOL2 READ 3
                 * ================================================= */

                S_P2_R3: begin

                    pool_temp = c2_q;

                    if (pool_temp > pool_max)
                        pool_max <= pool_temp;


                    state <= S_P2_SAVE;

                end


                /* =================================================
                 * MAXPOOL2 SAVE
                 * ================================================= */

                S_P2_SAVE: begin

                    if (pool_c == 6) begin

                        pool_c <= 0;


                        if (pool_r == 6) begin

                            pool_r <= 0;


                            if (pool_ch == 15) begin

                                pool_ch <= 0;

                                class_idx <= 0;
                                fc_feature <= 0;

                                accumulator <=
                                    {{56{fc_bias[0][7]}},
                                     fc_bias[0]};

                                best_value <=
                                    -64'sh7fffffffffffffff;

                                best_class <= 0;

                                state <= S_FC_ADDR;

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


                /* =================================================
                 * FC ADDRESS
                 * ================================================= */

                S_FC_ADDR: begin

                    p2_raddr <= fc_feature;

                    fcw_raddr <=
                        class_idx * 784 +
                        fc_feature;

                    state <= S_FC_MAC;

                end


                /* =================================================
                 * FC MAC
                 * ================================================= */

                S_FC_MAC: begin

                    product =
                        $signed(p2_q) *
                        $signed(fcw_q);

                    accumulator <=
                        accumulator + product;


                    if (fc_feature == 783) begin

                        state <= S_FC_SAVE;

                    end

                    else begin

                        fc_feature <= fc_feature + 1;

                    end

                end


                /* =================================================
                 * FC SAVE / ARGMAX
                 * ================================================= */

                S_FC_SAVE: begin

                    if ((class_idx == 0) ||
                        (accumulator > best_value)) begin

                        best_value <= accumulator;

                        best_class <=
                            class_idx[3:0];

                    end


                    if (class_idx == 9) begin

                        state <= S_ARG;

                    end

                    else begin

                        class_idx <= class_idx + 1;

                        fc_feature <= 0;

                        accumulator <=
                            {{56{fc_bias[class_idx+1][7]}},
                             fc_bias[class_idx+1]};

                        state <= S_FC_ADDR;

                    end

                end


                /* =================================================
                 * ARGMAX RESULT
                 * ================================================= */

                S_ARG: begin

                    predicted_class <= best_class;

                    state <= S_DONE;

                end


                /* =================================================
                 * DONE
                 * ================================================= */

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

