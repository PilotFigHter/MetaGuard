@echo off
setlocal

echo MetaGuard - Network Intrusion Detection
echo.

echo Stopping any existing server on port 4000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":4000" ^| findstr LISTENING') do (
    echo Killing process %%a...
    taskkill /F /PID %%a 2>nul
)

echo.
echo Starting MetaGuard...
echo Press Ctrl+C to stop
echo.

python MetaWeb\app.py

echo.
echo Cleaning up...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":4000" ^| findstr LISTENING') do (
    taskkill /F /PID %%a 2>nul
)

echo Done!