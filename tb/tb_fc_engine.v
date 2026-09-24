`timescale 1ns/1ps

module tb_fc_engine;

    reg clk;
    reg rst;
    reg start;

    wire done;
    wire [3:0] predicted_class;
    wire [3:0] class_count;

    cnn_fc_engine dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .done(done),
        .predicted_class(predicted_class),
        .class_count(class_count)
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

        $writememh("data/fc_output.mem", dut.fc_out);

        $display("");
        $display("====================================");
        $display("          FC ENGINE TEST");
        $display("====================================");

        $display("Class 0 = %0d", dut.fc_out[0]);
        $display("Class 1 = %0d", dut.fc_out[1]);
        $display("Class 2 = %0d", dut.fc_out[2]);
        $display("Class 3 = %0d", dut.fc_out[3]);
        $display("Class 4 = %0d", dut.fc_out[4]);
        $display("Class 5 = %0d", dut.fc_out[5]);
        $display("Class 6 = %0d", dut.fc_out[6]);
        $display("Class 7 = %0d", dut.fc_out[7]);
        $display("Class 8 = %0d", dut.fc_out[8]);
        $display("Class 9 = %0d", dut.fc_out[9]);

        $display("------------------------------------");
        $display("Classes calculated = %0d", class_count);
        $display("Predicted class    = %0d", predicted_class);
        $display("Expected class     = 7");

        if (predicted_class == 4'd7)
            $display("PASS");
        else
            $display("FAIL");

        $display("====================================");

        #100;
        $finish;

    end

endmodule
