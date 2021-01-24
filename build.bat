@echo off
rem 须先将 opencv.7z 解压到opencv/
rem 须先将 openvino.7z 解压到 openvino/
rem 须先将 mnn.7z 解压到 mnn/
rem 成功后 在顶层目录,执行 bin/vb.exe 等例子
if DEFINED _DOMAKE_ goto MAKE

rem 设置 opencv 环境
set OpenCV_DIR=%~dp0opencv\build

rem 查找 vs 安装目录
for /F "delims=" %%i in ('vswhere -property installationPath') do ( set installationPath=%%i )
if NOT DEFINED installationPath goto vc14

rem 并去掉尾空格
:intercept
if "%installationPath:~-1%"==" " set "installationPath=%installationPath:~0,-1%"&goto intercept

rem 查找 vs 版本号前2位
for /F "delims=" %%i in ('vswhere -property installationVersion') do ( set installationVersion=%%i )
set installationVersion=%installationVersion:~0,2%

rem 当前只支持 15
if "%installationVersion%" NEQ "15" goto ERROR

rem 设置 vs 环境
call "%installationPath%\Common7\Tools\VsDevCmd.bat" -arch=x64
goto InitEnv

:vc14
set installationVersion=14
call "%VS140COMNTOOLS%..\..\VC\vcvarsall.bat" x64


:InitEnv
set PATH=%OpenCV_DIR%\x64\vc%installationVersion%\bin;%PATH%

rem 设置 openvino 环境
set OpenVINO_DIR=%~dp0openvino\inference_engine
set PATH=%OpenVINO_DIR%\binary;%PATH%

rem 设置 mnn 环境
set MNN_DIR=%~dp0mnn
set PATH=%MNN_DIR%\x64\vc%installationVersion%\bin;%PATH%

:MAKE
if not exist build mkdir build
set _DOMAKE_=1
pushd build
rem cmake -G "Visual Studio 14 2015 Win64" ..
cmake -DCMAKE_BUILD_TYPE=release -G "NMake Makefiles" .. && nmake
popd

:ERROR
