`timescale 1ns/1ps

module tb_m10k_probe;

    reg clk = 0;
    reg rst = 1;
    reg start = 0;

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

    always #10 clk = ~clk;

    initial begin
        $display("========================================");
        $display(" M10K CNN INTERNAL DATA PROBE");
        $display("========================================");

        #100;
        rst = 0;

        #40;
        start = 1;
        #20;
        start = 0;

        wait(done);

        $display("");
        $display("Inference complete");
        $display("Cycles = %0d", cycle_count);
        $display("Class  = %0d", predicted_class);

        $display("");
        $display("---- Conv1 first 10 ----");
        $display("%h %h %h %h %h %h %h %h %h %h",
            dut.c1_ram.mem[0],
            dut.c1_ram.mem[1],
            dut.c1_ram.mem[2],
            dut.c1_ram.mem[3],
            dut.c1_ram.mem[4],
            dut.c1_ram.mem[5],
            dut.c1_ram.mem[6],
            dut.c1_ram.mem[7],
            dut.c1_ram.mem[8],
            dut.c1_ram.mem[9]);

        $display("");
        $display("---- Pool1 first 10 ----");
        $display("%h %h %h %h %h %h %h %h %h %h",
            dut.p1_ram.mem[0],
            dut.p1_ram.mem[1],
            dut.p1_ram.mem[2],
            dut.p1_ram.mem[3],
            dut.p1_ram.mem[4],
            dut.p1_ram.mem[5],
            dut.p1_ram.mem[6],
            dut.p1_ram.mem[7],
            dut.p1_ram.mem[8],
            dut.p1_ram.mem[9]);

        $display("");
        $display("---- Conv2 first 10 ----");
        $display("%h %h %h %h %h %h %h %h %h %h",
            dut.c2_ram.mem[0],
            dut.c2_ram.mem[1],
            dut.c2_ram.mem[2],
            dut.c2_ram.mem[3],
            dut.c2_ram.mem[4],
            dut.c2_ram.mem[5],
            dut.c2_ram.mem[6],
            dut.c2_ram.mem[7],
            dut.c2_ram.mem[8],
            dut.c2_ram.mem[9]);

        $display("");
        $display("---- Pool2 first 10 ----");
        $display("%h %h %h %h %h %h %h %h %h %h",
            dut.p2_ram.mem[0],
            dut.p2_ram.mem[1],
            dut.p2_ram.mem[2],
            dut.p2_ram.mem[3],
            dut.p2_ram.mem[4],
            dut.p2_ram.mem[5],
            dut.p2_ram.mem[6],
            dut.p2_ram.mem[7],
            dut.p2_ram.mem[8],
            dut.p2_ram.mem[9]);

        $display("");
        $display("---- FC results ----");
        $display("FC0 = %0d", dut.acc);
        $display("Best class = %0d", dut.best_class);

        $writememh("probe_conv1.mem", dut.c1_ram.mem);
        $writememh("probe_pool1.mem", dut.p1_ram.mem);
        $writememh("probe_conv2.mem", dut.c2_ram.mem);
        $writememh("probe_pool2.mem", dut.p2_ram.mem);

        $display("");
        $display("Probe files written.");
        $finish;
    end

    initial begin
        #40000000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
