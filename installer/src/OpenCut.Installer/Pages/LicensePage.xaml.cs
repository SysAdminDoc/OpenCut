using System.IO;
using System.Reflection;
using System.Windows;
using System.Windows.Controls;

namespace OpenCut.Installer.Pages;

public partial class LicensePage : Page
{
    private readonly MainWindow _mainWindow;

    public LicensePage(MainWindow mainWindow)
    {
        InitializeComponent();
        _mainWindow = mainWindow;
        Loaded += OnLoaded;
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        // Load embedded LICENSE.txt
        var assembly = Assembly.GetExecutingAssembly();
        var resourceName = assembly.GetManifestResourceNames()
            .FirstOrDefault(n => n.EndsWith("LICENSE.txt", StringComparison.OrdinalIgnoreCase));

        if (resourceName != null)
        {
            using var stream = assembly.GetManifestResourceStream(resourceName);
            if (stream != null)
            {
                using var reader = new StreamReader(stream);
                LicenseText.Text = reader.ReadToEnd();
            }
        }
        else
        {
            LicenseText.Text = "MIT License\n\nCopyright (c) 2025 OpenCut Contributors\n\n"
                + "Permission is hereby granted, free of charge, to any person obtaining a copy "
                + "of this software and associated documentation files (the \"Software\"), to deal "
                + "in the Software without restriction, including without limitation the rights "
                + "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell "
                + "copies of the Software, and to permit persons to whom the Software is "
                + "furnished to do so, subject to the following conditions:\n\n"
                + "The above copyright notice and this permission notice shall be included in all "
                + "copies or substantial portions of the Software.\n\n"
                + "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR "
                + "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, "
                + "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.";
        }
    }

    private void AcceptCheck_Changed(object sender, RoutedEventArgs e)
    {
        NextBtn.IsEnabled = AcceptCheck.IsChecked == true;
    }

    private void Back_Click(object sender, RoutedEventArgs e)
    {
        _mainWindow.SetStep(0);
        _mainWindow.NavigateToPage(new WelcomePage(_mainWindow));
    }

    private void Next_Click(object sender, RoutedEventArgs e)
    {
        _mainWindow.SetStep(2);
        _mainWindow.NavigateToPage(new OptionsPage(_mainWindow));
    }
}
