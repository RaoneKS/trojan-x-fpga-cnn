`timescale 1ns/1ps

/* Single-port synchronous RAM.  Quartus can infer Cyclone V M10K blocks. */
module cnn_sp_ram #(
    parameter WIDTH = 8,
    parameter DEPTH = 1,
    parameter AW = 1,
    parameter INIT_FILE = ""
)(
    input  wire         clk,
    input  wire         we,
    input  wire [AW-1:0] waddr,
    input  wire [WIDTH-1:0] wdata,
    input  wire [AW-1:0] raddr,
    output reg  [WIDTH-1:0] rdata
);
    (* ramstyle = "M10K" *) reg [WIDTH-1:0] mem [0:DEPTH-1];

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


module cnn_small_core (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    output reg         done,
    output reg [3:0]   predicted_class,
    output reg [63:0]  cycle_count
);

    localparam IDLE=6'd0,
               C1A=6'd1, C1W=6'd2, C1M=6'd3, C1S=6'd4, C1WR=6'd5,
               P1A=6'd6, P1W=6'd7, P1R0=6'd8, P1R1=6'd9, P1R2=6'd10, P1R3=6'd11, P1S=6'd12, P1WR=6'd13,
               C2A=6'd14, C2W=6'd15, C2M=6'd16, C2S=6'd17, C2WR=6'd18,
               P2A=6'd19, P2W=6'd20, P2R0=6'd21, P2R1=6'd22, P2R2=6'd23, P2R3=6'd24, P2S=6'd25, P2WR=6'd26,
               FCA=6'd27, FCW=6'd28, FCM=6'd29, FCS=6'd30,
               ARG=6'd31, FIN=6'd32;

    reg [5:0] state;

    /* Initialized input/weight RAMs */
    reg [9:0]  image_ra;
    reg [6:0]  c1w_ra;
    reg [10:0] c2w_ra;
    reg [12:0] fcw_ra;
    wire signed [7:0] image_q, c1w_q, c2w_q, fcw_q;

    cnn_sp_ram #(.WIDTH(8),.DEPTH(784),.AW(10),
                 .INIT_FILE("data/mnist_image0_int8.mem")) image_ram(
        .clk(clk),.we(1'b0),.waddr(10'd0),.wdata(8'd0),
        .raddr(image_ra),.rdata(image_q));

    cnn_sp_ram #(.WIDTH(8),.DEPTH(72),.AW(7),
                 .INIT_FILE("data/int8/conv1_weight.mem")) c1w_ram(
        .clk(clk),.we(1'b0),.waddr(7'd0),.wdata(8'd0),
        .raddr(c1w_ra),.rdata(c1w_q));

    cnn_sp_ram #(.WIDTH(8),.DEPTH(1152),.AW(11),
                 .INIT_FILE("data/int8/conv2_weight.mem")) c2w_ram(
        .clk(clk),.we(1'b0),.waddr(11'd0),.wdata(8'd0),
        .raddr(c2w_ra),.rdata(c2w_q));

    cnn_sp_ram #(.WIDTH(8),.DEPTH(7840),.AW(13),
                 .INIT_FILE("data/int8/fc1_weight.mem")) fcw_ram(
        .clk(clk),.we(1'b0),.waddr(13'd0),.wdata(8'd0),
        .raddr(fcw_ra),.rdata(fcw_q));

    reg signed [7:0] c1b[0:7];
    reg signed [7:0] c2b[0:15];
    reg signed [7:0] fcb[0:9];

    initial begin
        $readmemh("data/int8/conv1_bias.mem",c1b);
        $readmemh("data/int8/conv2_bias.mem",c2b);
        $readmemh("data/int8/fc1_bias.mem",fcb);
    end

    /*
     * Four large feature maps.  Each is a synchronous single-port RAM.
     * A separate write enable is used only in the corresponding SAVE state.
     */
    reg [12:0] c1_addr, c1_raddr;
    reg [10:0] p1_addr, p1_raddr;
    reg [11:0] c2_addr, c2_raddr;
    reg [9:0]  p2_addr, p2_raddr;

    reg signed [31:0] c1_wdata, p1_wdata, c2_wdata, p2_wdata;

    wire signed [31:0] c1_q, p1_q, c2_q, p2_q;

    cnn_sp_ram #(.WIDTH(32),.DEPTH(6272),.AW(13)) c1_ram(
        .clk(clk),.we(c1_we),.waddr(c1_addr),.wdata(c1_wdata),
        .raddr(c1_raddr),.rdata(c1_q));

    cnn_sp_ram #(.WIDTH(32),.DEPTH(1568),.AW(11)) p1_ram(
        .clk(clk),.we(p1_we),.waddr(p1_addr),.wdata(p1_wdata),
        .raddr(p1_raddr),.rdata(p1_q));

    cnn_sp_ram #(.WIDTH(32),.DEPTH(3136),.AW(12)) c2_ram(
        .clk(clk),.we(c2_we),.waddr(c2_addr),.wdata(c2_wdata),
        .raddr(c2_raddr),.rdata(c2_q));

    cnn_sp_ram #(.WIDTH(32),.DEPTH(784),.AW(10)) p2_ram(
        .clk(clk),.we(p2_we),.waddr(p2_addr),.wdata(p2_wdata),
        .raddr(p2_raddr),.rdata(p2_q));

    reg signed [63:0] acc;
    reg signed [63:0] product;

    integer out_r,out_c,out_f,in_ch,kr,kc;
    integer pool_r,pool_c,pool_ch;
    integer fc_feature,class_idx;
    integer ir,ic,addr,waddr;

    reg signed [31:0] pool_max;
    reg signed [31:0] tval;
    reg signed [63:0] fc_value;
    reg signed [63:0] best_value;
    reg [3:0] best_class;

    /* Write enables are asserted during dedicated WRITE states so that
       address/data registered in SAVE are present for the RAM write edge. */
    wire c1_we = (state == C1WR);
    wire p1_we = (state == P1WR);
    wire c2_we = (state == C2WR);
    wire p2_we = (state == P2WR);

    /*
     * The RAM read addresses are registered one clock before their values
     * are consumed.  Large arrays therefore remain memories instead of
     * becoming flip-flops.
     */
    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            predicted_class <= 0;
            cycle_count <= 0;
            image_ra <= 0; c1w_ra <= 0; c2w_ra <= 0; fcw_ra <= 0;
            c1_raddr <= 0; p1_raddr <= 0; c2_raddr <= 0; p2_raddr <= 0;
            out_r<=0; out_c<=0; out_f<=0; in_ch<=0; kr<=0; kc<=0;
            pool_r<=0; pool_c<=0; pool_ch<=0;
            fc_feature<=0; class_idx<=0;
            acc<=0; product<=0; pool_max<=0;
            best_value<=-64'sh7fffffffffffffff; best_class<=0;
        end else begin
            cycle_count <= cycle_count + 1'b1;

            case(state)

            IDLE: begin
                done <= 0;
                if(start) begin
                    out_r<=0; out_c<=0; out_f<=0; kr<=0; kc<=0;
                    acc <= {{56{c1b[0][7]}},c1b[0]};
                    state <= C1A;
                end
            end

            /* ---------- Conv1 ---------- */
            C1A: begin
                ir=out_r+kr-1; ic=out_c+kc-1;
                if(ir>=0 && ir<28 && ic>=0 && ic<28) begin
                    image_ra <= (ir*28+ic);
                    c1w_ra <= (out_f*9+kr*3+kc);
                end else begin
                    image_ra<=0; c1w_ra<=0;
                end
                state<=C1W;
            end

            C1W: begin
                state<=C1M;
            end

            C1M: begin
                ir=out_r+kr-1; ic=out_c+kc-1;
                if(ir>=0 && ir<28 && ic>=0 && ic<28) begin
                    product = $signed(image_q)*$signed(c1w_q);
                    acc <= acc+product;
                end
                if(kc==2) begin
                    kc<=0;
                    if(kr==2) begin kr<=0; state<=C1S; end
                    else kr<=kr+1;
                end else kc<=kc+1;
            end

            C1S: begin
                addr=out_f*784+out_r*28+out_c;
                c1_addr<=addr;
                if(acc<0) c1_wdata<=0; else c1_wdata<=acc[31:0];
                acc<=0;
                state<=C1WR;
            end

            C1WR: begin
                if(out_f==7) begin
                    out_f<=0;
                    if(out_c==27) begin
                        out_c<=0;
                        if(out_r==27) begin
                            out_r<=0; pool_r<=0; pool_c<=0; pool_ch<=0;
                            state<=P1A;
                        end else begin
                            out_r<=out_r+1;
                            acc<={{56{c1b[0][7]}},c1b[0]};
                            state<=C1A;
                        end
                    end else begin
                        out_c<=out_c+1;
                        acc<={{56{c1b[0][7]}},c1b[0]};
                        state<=C1A;
                    end
                end else begin
                    out_f<=out_f+1;
                    acc<={{56{c1b[out_f+1][7]}},c1b[out_f+1]};
                    state<=C1A;
                end
            end

            /*
             * Pool1: each RAM read is one cycle.  P1A establishes address 0,
             * P1R0..P1R3 consume the four returned values.
             */
            P1A: begin
                c1_raddr <= pool_ch*784+(pool_r*2)*28+(pool_c*2);
                pool_max <= -32'sh80000000;
                state<=P1W;
            end
            P1W: begin
                state<=P1R0;
            end
            P1R0: begin
                tval=c1_q;
                if(tval>pool_max) pool_max<=tval;
                c1_raddr<=pool_ch*784+(pool_r*2)*28+(pool_c*2)+1;
                state<=P1R1;
            end
            P1R1: begin
                tval=c1_q;
                if(tval>pool_max) pool_max<=tval;
                c1_raddr<=pool_ch*784+(pool_r*2+1)*28+(pool_c*2);
                state<=P1R2;
            end
            P1R2: begin
                tval=c1_q;
                if(tval>pool_max) pool_max<=tval;
                c1_raddr<=pool_ch*784+(pool_r*2+1)*28+(pool_c*2)+1;
                state<=P1R3;
            end
            P1R3: begin
                tval=c1_q;
                if(tval>pool_max) pool_max<=tval;
                state<=P1S;
            end
            P1S: begin
                addr=pool_ch*196+pool_r*14+pool_c;
                p1_addr<=addr;
                p1_wdata<=pool_max;
                state<=P1WR;
            end

            P1WR: begin
                if(pool_c==13) begin
                    pool_c<=0;
                    if(pool_r==13) begin
                        pool_r<=0;
                        if(pool_ch==7) begin
                            pool_ch<=0; out_r<=0; out_c<=0; out_f<=0;
                            in_ch<=0; kr<=0; kc<=0;
                            acc<={{56{c2b[0][7]}},c2b[0]};
                            state<=C2A;
                        end else begin
                            pool_ch<=pool_ch+1;
                            state<=P1A;
                        end
                    end else begin
                        pool_r<=pool_r+1;
                        state<=P1A;
                    end
                end else begin
                    pool_c<=pool_c+1;
                    state<=P1A;
                end
            end

            /* ---------- Conv2 ---------- */
            C2A: begin
                ir=out_r+kr-1; ic=out_c+kc-1;
                if(ir>=0 && ir<14 && ic>=0 && ic<14) begin
                    p1_raddr<=in_ch*196+ir*14+ic;
                    c2w_ra<=out_f*72+in_ch*9+kr*3+kc;
                end else begin
                    p1_raddr<=0; c2w_ra<=0;
                end
                state<=C2W;
            end

            C2W: begin
                state<=C2M;
            end

            C2M: begin
                ir=out_r+kr-1; ic=out_c+kc-1;
                if(ir>=0 && ir<14 && ic>=0 && ic<14) begin
                    product=$signed(p1_q)*$signed(c2w_q);
                    acc<=acc+product;
                end
                if(kc==2) begin
                    kc<=0;
                    if(kr==2) begin
                        kr<=0;
                        if(in_ch==7) begin in_ch<=0; state<=C2S; end
                        else in_ch<=in_ch+1;
                    end else kr<=kr+1;
                end else kc<=kc+1;
            end

            C2S: begin
                addr=out_f*196+out_r*14+out_c;
                c2_addr<=addr;
                c2_wdata<=(acc<0)?0:acc[31:0];
                acc<=0;
                state<=C2WR;
            end

            C2WR: begin
                if(out_f==15) begin
                    out_f<=0;
                    if(out_c==13) begin
                        out_c<=0;
                        if(out_r==13) begin
                            out_r<=0; pool_r<=0; pool_c<=0; pool_ch<=0;
                            state<=P2A;
                        end else begin
                            out_r<=out_r+1;
                            acc<={{56{c2b[0][7]}},c2b[0]};
                            state<=C2A;
                        end
                    end else begin
                        out_c<=out_c+1;
                        acc<={{56{c2b[0][7]}},c2b[0]};
                        state<=C2A;
                    end
                end else begin
                    out_f<=out_f+1;
                    acc<={{56{c2b[out_f+1][7]}},c2b[out_f+1]};
                    state<=C2A;
                end
            end

            /* ---------- Pool2 ---------- */
            P2A: begin
                c2_raddr<=pool_ch*196+(pool_r*2)*14+(pool_c*2);
                pool_max<=-32'sh80000000;
                state<=P2W;
            end
            P2W: begin
                state<=P2R0;
            end
            P2R0: begin
                tval=c2_q; if(tval>pool_max) pool_max<=tval;
                c2_raddr<=pool_ch*196+(pool_r*2)*14+(pool_c*2)+1;
                state<=P2R1;
            end
            P2R1: begin
                tval=c2_q; if(tval>pool_max) pool_max<=tval;
                c2_raddr<=pool_ch*196+(pool_r*2+1)*14+(pool_c*2);
                state<=P2R2;
            end
            P2R2: begin
                tval=c2_q; if(tval>pool_max) pool_max<=tval;
                c2_raddr<=pool_ch*196+(pool_r*2+1)*14+(pool_c*2)+1;
                state<=P2R3;
            end
            P2R3: begin
                tval=c2_q; if(tval>pool_max) pool_max<=tval;
                state<=P2S;
            end
            P2S: begin
                addr=pool_ch*49+pool_r*7+pool_c;
                p2_addr<=addr;
                p2_wdata<=pool_max;
                state<=P2WR;
            end

            P2WR: begin
                if(pool_c==6) begin
                    pool_c<=0;
                    if(pool_r==6) begin
                        pool_r<=0;
                        if(pool_ch==15) begin
                            pool_ch<=0; class_idx<=0; fc_feature<=0;
                            acc<={{56{fcb[0][7]}},fcb[0]};
                            state<=FCA;
                        end else begin
                            pool_ch<=pool_ch+1;
                            state<=P2A;
                        end
                    end else begin
                        pool_r<=pool_r+1;
                        state<=P2A;
                    end
                end else begin
                    pool_c<=pool_c+1;
                    state<=P2A;
                end
            end

            /* ---------- FC ---------- */
            FCA: begin
                p2_raddr<=fc_feature;
                fcw_ra<=class_idx*784+fc_feature;
                state<=FCW;
            end
            FCW: begin
                state<=FCM;
            end
            FCM: begin
                product=$signed(p2_q)*$signed(fcw_q);
                acc<=acc+product;
                if(fc_feature==783) state<=FCS;
                else fc_feature<=fc_feature+1;
            end
            FCS: begin
                /*
                 * FCM has already added feature 783 on the preceding clock.
                 * acc therefore contains the complete class score here.
                 */
                fc_value=acc;
                if(class_idx==0 || fc_value>best_value) begin
                    best_value<=fc_value;
                    best_class<=class_idx[3:0];
                end

                if(class_idx==9) begin
                    state<=ARG;
                end else begin
                    class_idx<=class_idx+1;
                    fc_feature<=0;
                    acc<={{56{fcb[class_idx+1][7]}},fcb[class_idx+1]};
                    state<=FCA;
                end
            end

            ARG: begin
                predicted_class<=best_class;
                state<=FIN;
            end

            FIN: begin
                done<=1'b1;
            end

            default: state<=IDLE;
            endcase
        end
    end
endmodule
