# ============================================================================
# run_synth.tcl — batch-synthesize each kernel for Zynq-7020 and export reports
#
# USAGE (from a shell with Vitis HLS on PATH):
#   vitis_hls -f run_synth.tcl
#
# It creates one HLS project + solution per kernel, runs C-synthesis (no board,
# CPU only), and leaves the report at:
#   projects/<kernel>/sol1/syn/report/<top>_csynth.xml
# Then run parse_hls_report.py to extract latency/II/DSP/BRAM into reports/*.json
#
# TARGET: xc7z020-clg400-1 (Zynq-7020, e.g. PYNQ-Z1 / Zedboard). 100 MHz (10 ns).
# ============================================================================

set PART   "xc7z020-clg400-1"
set PERIOD 10.0

# kernel_name  source_file            top_function
set kernels {
    conv3x3      kernels/conv3x3.cpp      conv3x3_top
    depthwise3x3 kernels/depthwise3x3.cpp depthwise3x3_top
    pointwise    kernels/pointwise.cpp    pointwise_top
    batchnorm    kernels/batchnorm.cpp    batchnorm_top
}

foreach {name src top} $kernels {
    puts "==== Synthesizing $name (top=$top) for $PART ===="
    open_project -reset projects/$name
    set_top $top
    add_files $src -cflags "-I kernels"
    open_solution -reset "sol1"
    set_part $PART
    create_clock -period $PERIOD -name default
    csynth_design
    close_project
}

puts "All kernels synthesized. Now run:  python parse_hls_report.py"
exit
