`timescale 1ns / 1ps

module fixed_rrelu #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 /*verilator public*/ = 32,
    parameter DATA_IN_0_PRECISION_1 /*verilator public*/ = 16,

    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 4,

    parameter DATA_IN_0_PARALLELISM_DIM_0 /*verilator public*/ = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 /*verilator public*/ = 1,

    parameter DATA_OUT_0_PRECISION_0 /*verilator public*/ = 32,
    parameter DATA_OUT_0_PRECISION_1 /*verilator public*/ = 16,

    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 0,

    parameter DATA_OUT_0_PARALLELISM_DIM_0  /*verilator public*/= 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 /*verilator public*/ = 1,

    parameter INPLACE     = 0,

    parameter UPPER       = 1, // from 1/2
    parameter LOWER       = 6, // to 1/16

    parameter LFSR_POLY   = 32'h04c11db7  // Default is Ethernet FCS Polynomial
    /* verilator lint_on UNUSEDPARAM */
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst_n,
    input clk,

    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    /* verilator lint_on UNUSEDSIGNAL */

    input  logic data_in_0_valid,
    output logic data_in_0_ready,

    output logic data_out_0_valid,
    input  logic data_out_0_ready
);

  localparam TOP_PREC   = 1 << DATA_IN_0_PRECISION_1;
  localparam UPPER_I    = TOP_PREC >> UPPER;
  localparam LOWER_I    = TOP_PREC >> LOWER;
  localparam LFSR_WIDTH = DATA_IN_0_PRECISION_0;
  localparam FRACTIONAL = DATA_OUT_0_PRECISION_0 - DATA_IN_0_PRECISION_1 - 1;
  localparam VALID_MASK = {1'b1, DATA_IN_0_PRECISION_1'(UPPER_I - 1)} & ~{1'b1, DATA_IN_0_PRECISION_1'(LOWER_I - 1)};
  localparam LFSR_MASK  = {{FRACTIONAL{1'b0}}, VALID_MASK};

  for (
      genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1; i++
  ) begin : RRelu 

    logic [LFSR_WIDTH-1:0] lfsr_state = '1; // Start at '1
    logic [LFSR_WIDTH-1:0] next_state;

    /* verilator lint_off UNUSEDSIGNAL */
    logic [2*DATA_IN_0_PRECISION_0-1:0] dout; 
    /* verilator lint_on UNUSEDSIGNAL */

    logic [DATA_IN_0_PRECISION_0-1:0] multiplier = next_state & LFSR_MASK;
    logic [DATA_OUT_0_PRECISION_0-1:0] adjusted_out = DATA_OUT_0_PRECISION_0'($signed(dout) >>> DATA_IN_0_PRECISION_1);

    always @(posedge clk) 
    begin
        lfsr_state <= next_state;
    end
    
    // You can find this module in mase/verilog-ethernet, or at https://github.com/alexforencich/verilog-lfsr/blob/master/rtl/lfsr.v
    lfsr #(
        .LFSR_WIDTH         (LFSR_WIDTH),
        .LFSR_POLY          (LFSR_POLY),
        .STYLE              ("AUTO")
    ) lfsr_I (

        // verilator lint_off PINCONNECTEMPTY 
        .data_in(),
        .data_out(),
        // verilator lint_on PINCONNECTEMPTY 
        
        .state_in(lfsr_state),

        .state_out(next_state)
    );

    assign dout = $signed(multiplier) * $signed(data_in_0[i]);
    
    always_comb
    begin : RReluOutput 
      if ($signed(data_in_0[i]) < 0)
      begin
        data_out_0[i] = adjusted_out; 
      end
      else 
      begin
        data_out_0[i] = data_in_0[i]; 
      end
    end
  end
  
  assign data_out_0_valid = data_in_0_valid;  
  assign data_in_0_ready  = data_out_0_ready;

endmodule

