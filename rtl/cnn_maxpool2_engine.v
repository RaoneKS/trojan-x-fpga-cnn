`timescale 1ns/1ps

module cnn_maxpool2_engine (
    input wire clk,
    input wire rst,
    input wire start,

    output reg done,
    output reg [9:0] output_count
);

    // Conv2: 14x14x16 = 3136
    reg signed [31:0] conv2 [0:3135];

    // Pool2: 7x7x16 = 784
    reg signed [31:0] pool2 [0:783];

    localparam IDLE = 2'd0;
    localparam CALC = 2'd1;
    localparam SAVE = 2'd2;
    localparam FINISH = 2'd3;

    reg [1:0] state;

    integer row;
    integer col;
    integer ch;

    integer addr0;
    integer addr1;
    integer addr2;
    integer addr3;
    integer out_addr;

    reg signed [31:0] max_value;

    initial begin
        $readmemh("data/conv2_output.mem", conv2);

        state = IDLE;
        done = 1'b0;
        output_count = 0;

        row = 0;
        col = 0;
        ch = 0;
    end

    always @(posedge clk) begin

        if (rst) begin
            state <= IDLE;
            done <= 1'b0;
            output_count <= 0;

            row <= 0;
            col <= 0;
            ch <= 0;
        end
        else begin

            case (state)

                IDLE: begin
                    done <= 1'b0;

                    if (start) begin
                        row <= 0;
                        col <= 0;
                        ch <= 0;
                        state <= CALC;
                    end
                end

                CALC: begin

                    addr0 = ch * 196 + (row * 2) * 14 + (col * 2);
                    addr1 = addr0 + 1;
                    addr2 = addr0 + 14;
                    addr3 = addr2 + 1;

                    max_value = conv2[addr0];

                    if (conv2[addr1] > max_value)
                        max_value = conv2[addr1];

                    if (conv2[addr2] > max_value)
                        max_value = conv2[addr2];

                    if (conv2[addr3] > max_value)
                        max_value = conv2[addr3];

                    state <= SAVE;
                end

                SAVE: begin

                    out_addr = ch * 49 + row * 7 + col;

                    pool2[out_addr] <= max_value;

                    output_count <= output_count + 1'b1;

                    if (ch == 15) begin

                        ch <= 0;

                        if (col == 6) begin

                            col <= 0;

                            if (row == 6) begin
                                state <= FINISH;
                            end
                            else begin
                                row <= row + 1;
                                state <= CALC;
                            end

                        end
                        else begin
                            col <= col + 1;
                            state <= CALC;
                        end

                    end
                    else begin
                        ch <= ch + 1;
                        state <= CALC;
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
