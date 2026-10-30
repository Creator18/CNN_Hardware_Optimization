@echo off
REM ==========================================================================
REM run_synth.bat - C-synthesize every design point used by the cost model,
REM for the Zynq-7020 (xc7z020-clg400-1) at 100 MHz.
REM
REM Each design point has a config hls_<name>.cfg and lands in work\<name>\,
REM which is exactly the set of directories parse_hls_report.py reads.
REM
REM USAGE: run from this hls\ directory in a normal cmd prompt:
REM   run_synth.bat
REM then:
REM   python parse_hls_report.py
REM   python fpga_constants_from_hls.py
REM
REM Requires Vitis 2026.1 (C-synthesis needs no board) and the free node-locked
REM license (Vivado Basic tier + Vitis HLS) at XILINXD_LICENSE_FILE.
REM ==========================================================================
setlocal

REM --- adjust this path if your install differs ---
set VITIS_SETTINGS=C:\AMDDesignTools\2026.1\Vitis\settings64.bat
call "%VITIS_SETTINGS%"

if "%XILINXD_LICENSE_FILE%"=="" set XILINXD_LICENSE_FILE=%USERPROFILE%\.Xilinx\Xilinx.lic

REM conv_tiled_*  : DSP-tiled conv sweep (TM x TN), the CONV cost source
REM dw_u1, dw_u2  : low-unroll depthwise points (depthwise is memory-bound)
REM fc_u1         : low-unroll fully-connected point
REM others        : default design point of each remaining operator
REM
REM `call` is required: v++ is v++.bat, and invoking it without call ends this
REM script after the first iteration.
for %%K in (conv_tiled_8x8 conv_tiled_16x8 conv_tiled_16x12 conv_tiled_16x16 dw_u1 dw_u2 depthwise3x3 pointwise batchnorm relu pool fc_u1 fc) do (
    echo ==== Synthesizing %%K ====
    call v++ -c --mode hls --config hls_%%K.cfg --work_dir work\%%K
)

REM Optional: the naive conv kernel (kernels\conv3x3.cpp) that exhibits the DSP
REM cliff discussed in the paper. Override IC_UNROLL to trace it, e.g.
REM   call v++ -c --mode hls --config hls_conv3x3.cfg --work_dir work\conv3x3_u2 ^
REM       --hls.syn.file_cflags "kernels/conv3x3.cpp,-I./kernels -DIC_UNROLL=2"

echo.
echo Synthesis done. Now run:  python parse_hls_report.py  then  python fpga_constants_from_hls.py
endlocal
