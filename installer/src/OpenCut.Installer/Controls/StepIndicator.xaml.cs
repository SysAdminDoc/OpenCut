using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace OpenCut.Installer.Controls;

public partial class StepIndicator : UserControl
{
    private readonly string[] _stepNames;
    private readonly List<Border> _stepDots = [];
    private readonly List<TextBlock> _stepLabels = [];
    private readonly List<Border> _connectors = [];
    private int _currentStep;

    public StepIndicator()
    {
        InitializeComponent();
        _stepNames = ["Welcome", "License", "Options", "Install", "Done"];
        BuildIndicator();
    }

    public int CurrentStep
    {
        get => _currentStep;
        set { _currentStep = value; UpdateVisuals(); }
    }

    private void BuildIndicator()
    {
        StepsPanel.Items.Clear();
        LabelsPanel.Children.Clear();
        _stepDots.Clear();
        _stepLabels.Clear();
        _connectors.Clear();

        for (int i = 0; i < _stepNames.Length; i++)
        {
            if (i > 0)
            {
                var line = new Border
                {
                    Width = 54,
                    Height = 2,
                    VerticalAlignment = VerticalAlignment.Center,
                    Background = FindBrush("Surface1"),
                    CornerRadius = new CornerRadius(999),
                    Margin = new Thickness(10, 0, 10, 0)
                };
                StepsPanel.Items.Add(line);
                _connectors.Add(line);
            }

            var number = new TextBlock
            {
                Text = (i + 1).ToString(),
                Foreground = FindBrush("Subtext1"),
                FontSize = 11,
                FontWeight = FontWeights.SemiBold,
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center
            };
            var dot = new Border
            {
                Width = 26,
                Height = 26,
                CornerRadius = new CornerRadius(13),
                Background = FindBrush("Surface1"),
                BorderBrush = FindBrush("Surface2"),
                BorderThickness = new Thickness(1),
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0),
                ToolTip = _stepNames[i],
                Child = number
            };
            StepsPanel.Items.Add(dot);
            _stepDots.Add(dot);

            var label = new TextBlock
            {
                Text = _stepNames[i],
                FontSize = 11.5,
                FontWeight = FontWeights.SemiBold,
                FontFamily = (FontFamily)(TryFindResource("UIFont") ?? new FontFamily("Segoe UI")),
                Foreground = FindBrush("Overlay1"),
                HorizontalAlignment = HorizontalAlignment.Center,
                TextAlignment = TextAlignment.Center
            };
            LabelsPanel.Children.Add(label);
            _stepLabels.Add(label);
        }
        UpdateVisuals();
    }

    private void UpdateVisuals()
    {
        for (int stepIndex = 0; stepIndex < _stepDots.Count; stepIndex++)
        {
            var dot = _stepDots[stepIndex];
            var label = _stepLabels[stepIndex];
            if (dot.Child is not TextBlock number) continue;

            if (stepIndex < _currentStep)
            {
                dot.Background = FindBrush("Green");
                dot.BorderBrush = FindBrush("Green");
                dot.Width = 26;
                dot.Height = 26;
                number.Text = "✓";
                number.Foreground = FindBrush("Crust");
                label.Foreground = FindBrush("Subtext1");
                label.FontWeight = FontWeights.SemiBold;
            }
            else if (stepIndex == _currentStep)
            {
                dot.Background = FindBrush("Blue");
                dot.BorderBrush = FindBrush("Lavender");
                dot.Width = 30;
                dot.Height = 30;
                number.Text = (stepIndex + 1).ToString();
                number.Foreground = FindBrush("Crust");
                label.Foreground = FindBrush("TextBrush");
                label.FontWeight = FontWeights.Bold;
            }
            else
            {
                dot.Background = FindBrush("Surface1");
                dot.BorderBrush = FindBrush("Surface2");
                dot.Width = 26;
                dot.Height = 26;
                number.Text = (stepIndex + 1).ToString();
                number.Foreground = FindBrush("Overlay1");
                label.Foreground = FindBrush("Overlay1");
                label.FontWeight = FontWeights.SemiBold;
            }
        }

        for (int connectorIndex = 0; connectorIndex < _connectors.Count; connectorIndex++)
        {
            _connectors[connectorIndex].Background = connectorIndex < _currentStep
                ? FindBrush("Green")
                : FindBrush("Surface1");
        }
    }

    private Brush FindBrush(string key)
    {
        return (Brush)(TryFindResource(key) ?? Brushes.Gray);
    }
}
