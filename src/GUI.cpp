#include "GUI.h"
#include "Common.h"

using namespace System;
using namespace System::Windows::Forms;
[STAThreadAttribute]

void main2(array<String^>^ args) {
	Application::SetCompatibleTextRenderingDefault(false);
	Application::EnableVisualStyles();
	
	CUDACoverageMaps::GUI gui;
	Application::Run(% gui);
}