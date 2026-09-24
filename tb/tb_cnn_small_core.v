`timescale 1ns/1ps

module tb_cnn_small_core;

    reg clk;
    reg rst;
    reg start;

    wire done;
    wire [3:0] predicted_class;
    wire [63:0] cycle_count;

    cnn_small_core dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .done(done),
        .predicted_class(predicted_class),
        .cycle_count(cycle_count)
    );

    /* 50 MHz clock */
    initial begin
        clk = 1'b0;
        forever #10 clk = ~clk;
    end

    initial begin

        rst = 1'b1;
        start = 1'b0;

        #100;

        rst = 1'b0;

        #40;

        start = 1'b1;

        #20;

        start = 1'b0;

        $display("");
        $display("========================================");
        $display(" CNN MNIST FULL INFERENCE TEST");
        $display("========================================");
        $display("");

        wait(done);

        #20;

        $display("Inference complete.");
        $display("Cycle count      = %0d", cycle_count);
        $display("Predicted class  = %0d", predicted_class);
        $display("Expected class   = 7");

        if (predicted_class == 4'd7) begin
            $display("");
            $display("========================================");
            $display(" PASS: CNN predicted digit 7");
            $display("========================================");
        end
        else begin
            $display("");
            $display("========================================");
            $display(" FAIL");
            $display("========================================");
        end

        $finish;

    end

    initial begin

        $dumpfile("cnn_full.vcd");
        $dumpvars(0, tb_cnn_small_core);

    end

    /*
     * Safety timeout.
     *
     * The complete sequential CNN should finish in
     * substantially less than this.
     */

    initial begin

        #20000000;

        $display("TIMEOUT");

        $finish;

    end

endmodule
