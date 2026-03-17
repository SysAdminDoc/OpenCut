using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using OpenCut.Installer.Models;

namespace OpenCut.Installer.Controls;

public partial class LogPanel : UserControl
{
    public LogPanel()
    {
        InitializeComponent();
    }

    public void AppendLog(string message, LogLevel level = LogLevel.Info)
    {
        var timestamp = DateTime.Now.ToString("HH:mm:ss");
        var prefix = level switch
        {
            LogLevel.Success => "[OK]",
            LogLevel.Warning => "[WARN]",
            LogLevel.Error => "[ERR]",
            LogLevel.Debug => "[DBG]",
            _ => "[...]"
        };

        var color = level switch
        {
            LogLevel.Success => FindColor("GreenColor"),
            LogLevel.Warning => FindColor("YellowColor"),
            LogLevel.Error => FindColor("RedColor"),
            LogLevel.Debug => FindColor("Overlay0Color"),
            _ => FindColor("Subtext0Color")
        };

        var paragraph = new Paragraph { Margin = new Thickness(0, 1, 0, 1) };

        var timeRun = new Run($"{timestamp} ")
        {
            Foreground = new SolidColorBrush(FindColor("Overlay0Color"))
        };
        var prefixRun = new Run($"{prefix} ")
        {
            Foreground = new SolidColorBrush(color),
            FontWeight = level == LogLevel.Error ? FontWeights.Bold : FontWeights.Normal
        };
        var messageRun = new Run(message)
        {
            Foreground = new SolidColorBrush(color)
        };

        paragraph.Inlines.Add(timeRun);
        paragraph.Inlines.Add(prefixRun);
        paragraph.Inlines.Add(messageRun);

        LogBox.Document.Blocks.Add(paragraph);
        LogBox.ScrollToEnd();
    }

    public void Clear()
    {
        LogBox.Document.Blocks.Clear();
    }

    private Color FindColor(string key)
    {
        if (TryFindResource(key) is Color c) return c;
        return Colors.Gray;
    }
}
