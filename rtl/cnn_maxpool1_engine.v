`timescale 1ns/1ps

module cnn_maxpool1_engine (
    input wire clk,
    input wire rst,
    input wire start,

    output reg done,
    output reg [11:0] output_count
);

    // Conv1 feature map: 28 x 28 x 8
    reg signed [31:0] conv1 [0:6271];

    // MaxPool output: 14 x 14 x 8
    reg signed [31:0] pool1 [0:1567];

    reg [2:0] state;

    localparam IDLE = 3'd0;
    localparam CALC = 3'd1;
    localparam SAVE = 3'd2;
    localparam DONE = 3'd3;

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
        // Load the Conv1 reference output.
        $readmemh("data/conv1_output.mem", conv1);

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

                    addr0 = ch * 784 + (row * 2) * 28 + (col * 2);
                    addr1 = addr0 + 1;
                    addr2 = addr0 + 28;
                    addr3 = addr2 + 1;

                    max_value = conv1[addr0];

                    if (conv1[addr1] > max_value)
                        max_value = conv1[addr1];

                    if (conv1[addr2] > max_value)
                        max_value = conv1[addr2];

                    if (conv1[addr3] > max_value)
                        max_value = conv1[addr3];

                    state <= SAVE;
                end

                SAVE: begin

                    out_addr = ch * 196 + row * 14 + col;

                    pool1[out_addr] <= max_value;

                    output_count <= output_count + 1'b1;

                    if (ch == 7) begin

                        ch <= 0;

                        if (col == 13) begin

                            col <= 0;

                            if (row == 13) begin
                                state <= DONE;
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
