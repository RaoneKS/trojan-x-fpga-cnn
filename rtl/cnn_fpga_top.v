`timescale 1ns/1ps

module cnn_fpga_top (
    input  wire       CLOCK_50,
    input  wire [0:0] KEY,
    output wire [3:0] LEDR
);

    wire rst;
    reg start;
    reg started;

    wire done;
    wire [3:0] predicted_class;
    wire [63:0] cycle_count;

    assign rst = ~KEY[0];

    always @(posedge CLOCK_50) begin

        if (rst) begin
            start <= 1'b0;
            started <= 1'b0;
        end
        else begin

            if (!started) begin
                start <= 1'b1;
                started <= 1'b1;
            end
            else begin
                start <= 1'b0;
            end

        end
    end

    cnn_small_core u_core (
        .clk(CLOCK_50),
        .rst(rst),
        .start(start),
        .done(done),
        .predicted_class(predicted_class),
        .cycle_count(cycle_count)
    );

    assign LEDR = predicted_class;

endmodule
