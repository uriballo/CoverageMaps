#include "GUI.h"
#include "src/Common.h"

using namespace System;
using namespace System::Windows::Forms;
[STAThreadAttribute]

void main(array<String^>^ args) {
	Application::SetCompatibleTextRenderingDefault(false);
	Application::EnableVisualStyles();
	
	CUDACoverageMaps::GUI gui;
	Application::Run(% gui);
}