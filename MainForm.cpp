#include "MainForm.h"
#include "src/Common.h"

using namespace System;
using namespace System::Windows::Forms;
[STAThreadAttribute]

void main(array<String^>^ args) {
	Application::SetCompatibleTextRenderingDefault(false);
	Application::EnableVisualStyles();

	CUDACoverageMaps::MainForm gui;
	Application::Run(% gui);
}

