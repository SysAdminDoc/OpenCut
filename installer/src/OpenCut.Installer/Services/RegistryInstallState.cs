using Microsoft.Win32;

namespace OpenCut.Installer.Services;

public sealed record RegistryValueSnapshot(object Value, RegistryValueKind Kind);

public sealed record RegistryKeySnapshot(
    bool Exists,
    IReadOnlyDictionary<string, RegistryValueSnapshot> Values);

public sealed class RegistryInstallState
{
    public required RegistryKeySnapshot UserEnvironment { get; init; }
    public required RegistryKeySnapshot UserApplication { get; init; }
    public required RegistryKeySnapshot MachineUninstall { get; init; }
    public required IReadOnlyDictionary<int, RegistryValueSnapshot?> PlayerDebugModes { get; init; }
}
