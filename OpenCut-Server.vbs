Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

strPath = fso.GetParentFolderName(WScript.ScriptFullName)

' Build environment
strPython = "python"
strEnv = "set OPENCUT_HOME=" & strPath

If fso.FileExists(strPath & "\python\python.exe") Then
    strPython = """" & strPath & "\python\python.exe"""
    strEnv = strEnv & " & set PATH=" & strPath & "\python;" & strPath & "\python\Scripts;%PATH%"
End If

If fso.FolderExists(strPath & "\ffmpeg") Then
    strEnv = strEnv & " & set PATH=" & strPath & "\ffmpeg;%PATH%"
End If

If fso.FolderExists(strPath & "\models") Then
    strEnv = strEnv & " & set OPENCUT_BUNDLED=true"
    strEnv = strEnv & " & set WHISPER_MODELS_DIR=" & strPath & "\models\whisper"
    strEnv = strEnv & " & set TORCH_HOME=" & strPath & "\models\demucs"
    strEnv = strEnv & " & set OPENCUT_FLORENCE_DIR=" & strPath & "\models\florence"
    strEnv = strEnv & " & set OPENCUT_LAMA_DIR=" & strPath & "\models\lama"
End If

strCmd = "cmd /c """ & strEnv & " & " & strPython & " -m opencut.server"""

' Run completely hidden (0 = hidden, False = don't wait)
WshShell.Run strCmd, 0, False
