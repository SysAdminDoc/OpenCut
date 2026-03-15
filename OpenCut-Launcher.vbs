Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

strPath = fso.GetParentFolderName(WScript.ScriptFullName)
strExe = """" & strPath & "\server\OpenCut-Server.exe"""

' Add bundled ffmpeg to PATH if present
strEnv = ""
If fso.FolderExists(strPath & "\server\ffmpeg") Then
    strEnv = "set PATH=" & strPath & "\server\ffmpeg;%PATH% & "
End If

strCmd = "cmd /c """ & strEnv & strExe & """"

' Run completely hidden (0 = hidden, False = don't wait)
WshShell.Run strCmd, 0, False
