`timescale 1ns/1ps

module tb_maxpool1_engine;

    reg clk;
    reg rst;
    reg start;

    wire done;
    wire [11:0] output_count;

    cnn_maxpool1_engine dut (
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

        $writememh("data/pool1_output.mem", dut.pool1);
        $display("");
        $display("====================================");
        $display("       MAXPOOL1 ENGINE TEST");
        $display("====================================");
        $display("Pool1 outputs = %0d", output_count);
        $display("Expected      = 1568");

        if (output_count == 1568)
            $display("PASS");
        else
            $display("FAIL");

        $display("====================================");

        #100;
        $finish;

    end

endmodule
