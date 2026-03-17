namespace OpenCut.Installer.Models;

public enum LogLevel
{
    Info,
    Success,
    Warning,
    Error,
    Debug
}

public class InstallProgress
{
    public int StepNumber { get; init; }
    public int TotalSteps { get; init; }
    public string StepName { get; init; } = "";
    public string Message { get; init; } = "";
    public LogLevel Level { get; init; } = LogLevel.Info;
    public double OverallPercent => TotalSteps > 0 ? (double)StepNumber / TotalSteps * 100 : 0;
}
