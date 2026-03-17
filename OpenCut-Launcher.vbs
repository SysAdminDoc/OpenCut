Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

strPath = fso.GetParentFolderName(WScript.ScriptFullName)
strExe = strPath & "\server\OpenCut-Server.exe"

' Add bundled ffmpeg to PATH if present
Dim objEnv
Set objEnv = WshShell.Environment("Process")
If fso.FolderExists(strPath & "\ffmpeg") Then
    objEnv("PATH") = strPath & "\ffmpeg;" & objEnv("PATH")
ElseIf fso.FolderExists(strPath & "\server\ffmpeg") Then
    objEnv("PATH") = strPath & "\server\ffmpeg;" & objEnv("PATH")
End If

' Run completely hidden (0 = hidden, False = don't wait)
WshShell.Run """" & strExe & """", 0, False
