@echo off
REM ==========================================================================
REM run_synth.bat - synthesize each HLS kernel for Zynq-7020 via the v++ flow.
REM
REM Vivado/Vitis 2026.1: HLS C-synthesis is driven by `v++ -c --mode hls` with
REM a per-kernel .cfg (the classic `vitis_hls -f script.tcl` wrapper was removed
REM and its Tcl path handling is broken under vitis-run). Reports land in
REM   work\<kernel>\hls\syn\report\<top>_csynth.xml
REM
REM USAGE: run from this hls\ directory in a normal cmd prompt:
REM   run_synth.bat
REM Requires XILINXD_LICENSE_FILE to point at your Xilinx.lic (free Basic tier).
REM ==========================================================================
setlocal

REM --- adjust this path if your install differs ---
set VITIS_SETTINGS=C:\AMDDesignTools\2026.1\Vitis\settings64.bat
call "%VITIS_SETTINGS%"

if "%XILINXD_LICENSE_FILE%"=="" set XILINXD_LICENSE_FILE=%USERPROFILE%\.Xilinx\Xilinx.lic

for %%K in (conv3x3 depthwise3x3 pointwise batchnorm) do (
    echo ==== Synthesizing %%K ====
    v++ -c --mode hls --config hls_%%K.cfg --work_dir work\%%K
)

echo.
echo Synthesis done. Now run:  python parse_hls_report.py  ^&^&  python fpga_constants_from_hls.py
endlocal
