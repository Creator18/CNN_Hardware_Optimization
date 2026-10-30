@echo off
REM ==========================================================================
REM run_impl.bat - validate the chosen tiled conv (TM=16, TN=8) beyond
REM C-synthesis:
REM   1. C-simulation against a reference convolution (tb\tb_conv.cpp)
REM   2. Vivado out-of-context synthesis + place-and-route on xc7z020
REM Results are summarized in reports\conv_impl.json.
REM
REM USAGE: run from this hls\ directory:  run_impl.bat
REM
REM Implementation runs in a SHORT directory (IMPL_DIR) on purpose: Vitis
REM generates deeply nested IP paths with long mangled names, and a work dir
REM under a long project path exceeds the Windows 260-character path limit.
REM ==========================================================================
setlocal

set VITIS_SETTINGS=C:\AMDDesignTools\2026.1\Vitis\settings64.bat
call "%VITIS_SETTINGS%"
if "%XILINXD_LICENSE_FILE%"=="" set XILINXD_LICENSE_FILE=%USERPROFILE%\.Xilinx\Xilinx.lic

set HLS_DIR=%~dp0
set IMPL_DIR=C:\hlsimpl

echo ==== 1. C-simulation (expects: PASS on all 200704 outputs) ====
call vitis-run --mode hls --csim --config hls_conv_tiled_csim.cfg --work_dir work\conv_csim

echo ==== 2. C-synthesis + implementation (place-and-route) ====
if exist "%IMPL_DIR%" rmdir /s /q "%IMPL_DIR%"
mkdir "%IMPL_DIR%"
call v++ -c --mode hls --config "%HLS_DIR%hls_conv_tiled_16x8.cfg" --work_dir "%IMPL_DIR%\w"
call vitis-run --mode hls --impl --config "%HLS_DIR%hls_conv_tiled_16x8.cfg" --work_dir "%IMPL_DIR%\w"

echo.
echo Look for "Post-Implementation Resource usage" and "CP achieved post-implementation"
echo in the output above (or under %IMPL_DIR%\w).
endlocal
