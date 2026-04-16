Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

strPath = fso.GetParentFolderName(WScript.ScriptFullName)
strExe = strPath & "\server\OpenCut-Server.exe"

' Verify the server exe exists before trying to launch it. Without this
' guard the WScript.Run call below silently fails when the install is
' broken or the user moved the launcher out of the install dir, leaving
' the user wondering why nothing happens.
If Not fso.FileExists(strExe) Then
    MsgBox "OpenCut server executable not found:" & vbCrLf & vbCrLf & strExe & vbCrLf & vbCrLf & _
           "The OpenCut installation appears to be missing or corrupt. Please reinstall.", _
           vbCritical + vbOKOnly, "OpenCut"
    WScript.Quit 1
End If

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
