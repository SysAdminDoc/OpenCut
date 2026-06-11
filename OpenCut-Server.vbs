Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
Set env = WshShell.Environment("Process")

strPath = fso.GetParentFolderName(WScript.ScriptFullName)

' Build environment via WshShell.Environment (safe for paths with & ^ etc.)
env("OPENCUT_HOME") = strPath

strPython = "python"
If fso.FileExists(strPath & "\python\python.exe") Then
    strPython = """" & strPath & "\python\python.exe"""
    env("PATH") = strPath & "\python;" & strPath & "\python\Scripts;" & env("PATH")
End If

If fso.FolderExists(strPath & "\ffmpeg") Then
    env("PATH") = strPath & "\ffmpeg;" & env("PATH")
End If

If fso.FolderExists(strPath & "\models") Then
    env("OPENCUT_BUNDLED") = "true"
    env("WHISPER_MODELS_DIR") = strPath & "\models\whisper"
    env("TORCH_HOME") = strPath & "\models\demucs"
    env("OPENCUT_FLORENCE_DIR") = strPath & "\models\florence"
    env("OPENCUT_LAMA_DIR") = strPath & "\models\lama"
End If

' Run completely hidden (0 = hidden, False = don't wait)
WshShell.Run strPython & " -m opencut.server", 0, False
