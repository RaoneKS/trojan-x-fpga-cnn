`timescale 1ns/1ps

module tb_conv2_engine;

    reg clk;
    reg rst;
    reg start;

    wire done;
    wire [11:0] output_count;

    cnn_conv2_engine dut (
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

        $writememh("data/conv2_output.mem", dut.conv2);

        $display("");
        $display("====================================");
        $display("        CONV2 ENGINE TEST");
        $display("====================================");
        $display("Conv2 outputs = %0d", output_count);
        $display("Expected      = 3136");

        if (output_count == 3136)
            $display("PASS");
        else
            $display("FAIL");

        $display("====================================");

        #100;
        $finish;

    end

endmodule
