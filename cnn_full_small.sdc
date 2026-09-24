create_clock -name CLOCK_50 -period 20.000 [get_ports {CLOCK_50}]

set_input_delay -clock CLOCK_50 0.0 [get_ports {KEY[0]}]

set_output_delay -clock CLOCK_50 0.0 [get_ports {LEDR[*]}]
