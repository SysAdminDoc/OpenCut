using System.Runtime.InteropServices;

namespace OpenCut.Installer;

public static partial class NativeMethods
{
    public const int HWND_BROADCAST = 0xFFFF;
    public const int WM_SETTINGCHANGE = 0x001A;

    [LibraryImport("user32.dll", SetLastError = true, StringMarshalling = StringMarshalling.Utf16)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static partial bool SendMessageTimeoutW(
        nint hWnd,
        uint Msg,
        nint wParam,
        string lParam,
        uint fuFlags,
        uint uTimeout,
        out nint lpdwResult);

    public static void BroadcastEnvironmentChange()
    {
        SendMessageTimeoutW(
            (nint)HWND_BROADCAST,
            WM_SETTINGCHANGE,
            0,
            "Environment",
            0x0002, // SMTO_ABORTIFHUNG
            5000,
            out _);
    }
}
