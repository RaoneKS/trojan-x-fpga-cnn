`timescale 1ns/1ps

module tb_maxpool2_engine;

    reg clk;
    reg rst;
    reg start;

    wire done;
    wire [9:0] output_count;

    cnn_maxpool2_engine dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .done(done),
        .output_count(output_count)
    );

    always #10 clk = ~clk;

    initial begin

        clk = 0;
        rst = 1;
        start = 0;

        #100;

        rst = 0;

        #40;
        start = 1;

        #20;
        start = 0;

        wait(done);

        $writememh("data/pool2_output.mem", dut.pool2);

        $display("");
        $display("====================================");
        $display("       MAXPOOL2 ENGINE TEST");
        $display("====================================");
        $display("Pool2 outputs = %0d", output_count);
        $display("Expected      = 784");

        if (output_count == 784)
            $display("PASS");
        else
            $display("FAIL");

        $display("====================================");

        #100;
        $finish;

    end

endmodule
