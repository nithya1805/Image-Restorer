@echo off
REM Run ONCE as Administrator (right-click -> Run as administrator).
REM Lets devices on the local network (same subnet only) reach the app on TCP port 5003.
netsh advfirewall firewall delete rule name="Image Quality Checker (TCP 5003)" >nul 2>&1
netsh advfirewall firewall add rule name="Image Quality Checker (TCP 5003)" dir=in action=allow protocol=TCP localport=5003 remoteip=localsubnet profile=any
pause
