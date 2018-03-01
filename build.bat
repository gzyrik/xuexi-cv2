@echo off

if DEFINED OpenCV_DIR goto MAKE
set OpenCV_DIR=%~dp0opencv\build
call "%VS140COMNTOOLS%..\..\VC\vcvarsall.bat" x64
set PATH=%OpenCV_DIR%\x64\vc14\bin;%PATH%

:MAKE
if not exist build mkdir build
pushd build
rem cmake -G "Visual Studio 14 2015 Win64" ..
cmake -DCMAKE_BUILD_TYPE=release -G "NMake Makefiles" .. && nmake
popd