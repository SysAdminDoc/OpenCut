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

' Reject unsupported source runtimes before importing OpenCut. The packaged
' OpenCut-Launcher.vbs starts the bundled server executable and does not use
' the host Python installation.
On Error Resume Next
Set versionCheck = WshShell.Exec(strPython & _
    " -c ""import sys; print('.'.join(map(str, sys.version_info[:3]))); " & _
    "raise SystemExit(0 if sys.version_info >= (3, 11) else 1)""")
If Err.Number <> 0 Then
    Err.Clear
    On Error GoTo 0
    MsgBox "OpenCut requires Python 3.11 or later, but Python could not be started." & _
           vbCrLf & vbCrLf & "Install a supported version from:" & vbCrLf & _
           "https://www.python.org/downloads/", _
           vbCritical + vbOKOnly, "OpenCut"
    WScript.Quit 1
End If
On Error GoTo 0

detectedPython = Trim(versionCheck.StdOut.ReadAll)
Do While versionCheck.Status = 0
    WScript.Sleep 10
Loop
If versionCheck.ExitCode <> 0 Then
    If detectedPython = "" Then detectedPython = "unavailable"
    MsgBox "Detected Python " & detectedPython & "; OpenCut requires Python 3.11 or later." & _
           vbCrLf & vbCrLf & "Install a supported version from:" & vbCrLf & _
           "https://www.python.org/downloads/", _
           vbCritical + vbOKOnly, "OpenCut"
    WScript.Quit 1
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
