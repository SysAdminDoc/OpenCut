using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace OpenCut.Installer.Controls;

public partial class StepIndicator : UserControl
{
    private readonly string[] _stepNames;
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
        for (int i = 0; i < _stepNames.Length; i++)
        {
            if (i > 0)
            {
                var line = new Rectangle
                {
                    Width = 32,
                    Height = 2,
                    Fill = FindBrush("Surface1"),
                    VerticalAlignment = VerticalAlignment.Center,
                    Margin = new Thickness(0)
                };
                StepsPanel.Items.Add(line);
            }

            var dot = new Ellipse
            {
                Width = 10,
                Height = 10,
                Fill = FindBrush("Surface1"),
                Stroke = FindBrush("Surface2"),
                StrokeThickness = 1,
                VerticalAlignment = VerticalAlignment.Center,
                ToolTip = _stepNames[i]
            };
            StepsPanel.Items.Add(dot);
        }
        UpdateVisuals();
    }

    private void UpdateVisuals()
    {
        int itemIndex = 0;
        int stepIndex = 0;
        foreach (var item in StepsPanel.Items)
        {
            if (item is Ellipse dot)
            {
                if (stepIndex < _currentStep)
                {
                    dot.Fill = FindBrush("Green");
                    dot.Stroke = FindBrush("Green");
                }
                else if (stepIndex == _currentStep)
                {
                    dot.Fill = FindBrush("Blue");
                    dot.Stroke = FindBrush("Blue");
                    dot.Width = 12;
                    dot.Height = 12;
                }
                else
                {
                    dot.Fill = FindBrush("Surface1");
                    dot.Stroke = FindBrush("Surface2");
                    dot.Width = 10;
                    dot.Height = 10;
                }
                stepIndex++;
            }
            else if (item is Rectangle line)
            {
                line.Fill = stepIndex <= _currentStep ? FindBrush("Green") : FindBrush("Surface1");
            }
            itemIndex++;
        }
    }

    private Brush FindBrush(string key)
    {
        return (Brush)(TryFindResource(key) ?? Brushes.Gray);
    }
}
