#pragma once
#include "src/EuclideanExpansion.h"
#include "src/FeedbackExpansion.h"
#include "src/ImageDisplay.h"
#include <msclr/marshal_cppstd.h>
#include <string>
#include <sstream>

using namespace msclr::interop;

namespace CUDACoverageMaps {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for MainForm
	/// </summary>
	public ref class MainForm : public System::Windows::Forms::Form
	{
	public:
		MainForm(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
			customDistributionTB->Text = "x1,y1,x2,y2,x3,y3...";
			iterations = 1;
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~MainForm()
		{
			if (components)
			{
				delete components;
			}
		}

	protected:
	private: System::Windows::Forms::Label^ titleLabel;






	private: System::Windows::Forms::TableLayoutPanel^ tableLayoutPanel1;
	private: System::Windows::Forms::Button^ exitButton;
	private: System::Windows::Forms::Button^ runButton;
	private: System::Windows::Forms::OpenFileDialog^ openFileDialog1;
	private: System::Windows::Forms::NumericUpDown^ numberOfServicesNum;
	private: System::Windows::Forms::Label^ numServicesLabel;


	private: System::Windows::Forms::NumericUpDown^ serviceRadiusNum;
	private: System::Windows::Forms::Label^ serviceCoverageRadiusLabel;





	private: System::Windows::Forms::CheckBox^ customDistributionCB;
	private: System::Windows::Forms::TextBox^ customDistributionTB;

	private: System::Windows::Forms::CheckBox^ storeBoundary;
	private: System::Windows::Forms::CheckBox^ storeIterations;




	private: System::Windows::Forms::Label^ inputLabel;


	private: System::Windows::Forms::CheckBox^ maximumCoverageCB;


	private: System::Windows::Forms::CheckBox^ exactExpansionCB;

	private: System::Windows::Forms::CheckBox^ euclideanExpansionCB;

	private: System::Windows::Forms::Label^ serviceConfigLabel;
	private: System::Windows::Forms::Label^ outputLabel;


	private: System::Windows::Forms::Label^ algorithmConfigLabel;
	private: System::Windows::Forms::Label^ metaheuristicsLabel;
	private: System::Windows::Forms::Label^ solutionDataLabel;
	private: System::Windows::Forms::RichTextBox^ solutionDataRTB;
	private: System::Windows::Forms::Button^ openImageBT;

	private: System::Windows::Forms::TextBox^ inputPathTB;






























	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->inputPathTB = (gcnew System::Windows::Forms::TextBox());
			this->titleLabel = (gcnew System::Windows::Forms::Label());
			this->tableLayoutPanel1 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->exitButton = (gcnew System::Windows::Forms::Button());
			this->runButton = (gcnew System::Windows::Forms::Button());
			this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->numberOfServicesNum = (gcnew System::Windows::Forms::NumericUpDown());
			this->numServicesLabel = (gcnew System::Windows::Forms::Label());
			this->serviceRadiusNum = (gcnew System::Windows::Forms::NumericUpDown());
			this->serviceCoverageRadiusLabel = (gcnew System::Windows::Forms::Label());
			this->customDistributionCB = (gcnew System::Windows::Forms::CheckBox());
			this->customDistributionTB = (gcnew System::Windows::Forms::TextBox());
			this->storeBoundary = (gcnew System::Windows::Forms::CheckBox());
			this->storeIterations = (gcnew System::Windows::Forms::CheckBox());
			this->inputLabel = (gcnew System::Windows::Forms::Label());
			this->maximumCoverageCB = (gcnew System::Windows::Forms::CheckBox());
			this->exactExpansionCB = (gcnew System::Windows::Forms::CheckBox());
			this->euclideanExpansionCB = (gcnew System::Windows::Forms::CheckBox());
			this->serviceConfigLabel = (gcnew System::Windows::Forms::Label());
			this->outputLabel = (gcnew System::Windows::Forms::Label());
			this->algorithmConfigLabel = (gcnew System::Windows::Forms::Label());
			this->metaheuristicsLabel = (gcnew System::Windows::Forms::Label());
			this->solutionDataLabel = (gcnew System::Windows::Forms::Label());
			this->solutionDataRTB = (gcnew System::Windows::Forms::RichTextBox());
			this->openImageBT = (gcnew System::Windows::Forms::Button());
			this->tableLayoutPanel1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numberOfServicesNum))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->serviceRadiusNum))->BeginInit();
			this->SuspendLayout();
			// 
			// inputPathTB
			// 
			this->inputPathTB->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->inputPathTB->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->inputPathTB->Location = System::Drawing::Point(67, 123);
			this->inputPathTB->MaxLength = 60;
			this->inputPathTB->Name = L"inputPathTB";
			this->inputPathTB->Size = System::Drawing::Size(391, 34);
			this->inputPathTB->TabIndex = 21;
			this->inputPathTB->Text = L"N/A";
			// 
			// titleLabel
			// 
			this->titleLabel->AutoSize = true;
			this->titleLabel->Location = System::Drawing::Point(42, 25);
			this->titleLabel->Name = L"titleLabel";
			this->titleLabel->Size = System::Drawing::Size(334, 45);
			this->titleLabel->TabIndex = 1;
			this->titleLabel->Text = L"CUDA Coverage Maps";
			// 
			// tableLayoutPanel1
			// 
			this->tableLayoutPanel1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left)
				| System::Windows::Forms::AnchorStyles::Right));
			this->tableLayoutPanel1->ColumnCount = 2;
			this->tableLayoutPanel1->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
				50)));
			this->tableLayoutPanel1->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
				50)));
			this->tableLayoutPanel1->Controls->Add(this->exitButton, 0, 0);
			this->tableLayoutPanel1->Controls->Add(this->runButton, 1, 0);
			this->tableLayoutPanel1->Location = System::Drawing::Point(12, 755);
			this->tableLayoutPanel1->Name = L"tableLayoutPanel1";
			this->tableLayoutPanel1->RowCount = 1;
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 50)));
			this->tableLayoutPanel1->Size = System::Drawing::Size(620, 42);
			this->tableLayoutPanel1->TabIndex = 7;
			// 
			// exitButton
			// 
			this->exitButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left)
				| System::Windows::Forms::AnchorStyles::Right));
			this->exitButton->Location = System::Drawing::Point(3, 3);
			this->exitButton->Name = L"exitButton";
			this->exitButton->Size = System::Drawing::Size(304, 36);
			this->exitButton->TabIndex = 0;
			this->exitButton->Text = L"Exit";
			this->exitButton->UseVisualStyleBackColor = true;
			this->exitButton->Click += gcnew System::EventHandler(this, &MainForm::exitButton_Click);
			// 
			// runButton
			// 
			this->runButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left)
				| System::Windows::Forms::AnchorStyles::Right));
			this->runButton->Location = System::Drawing::Point(313, 3);
			this->runButton->Name = L"runButton";
			this->runButton->Size = System::Drawing::Size(304, 36);
			this->runButton->TabIndex = 1;
			this->runButton->Text = L"Run";
			this->runButton->UseVisualStyleBackColor = true;
			this->runButton->Click += gcnew System::EventHandler(this, &MainForm::runButton_Click);
			// 
			// openFileDialog1
			// 
			this->openFileDialog1->FileName = L"openFileDialog1";
			// 
			// numberOfServicesNum
			// 
			this->numberOfServicesNum->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->numberOfServicesNum->Location = System::Drawing::Point(64, 292);
			this->numberOfServicesNum->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 20, 0, 0, 0 });
			this->numberOfServicesNum->Name = L"numberOfServicesNum";
			this->numberOfServicesNum->Size = System::Drawing::Size(101, 34);
			this->numberOfServicesNum->TabIndex = 0;
			this->numberOfServicesNum->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			// 
			// numServicesLabel
			// 
			this->numServicesLabel->AutoSize = true;
			this->numServicesLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->numServicesLabel->Location = System::Drawing::Point(175, 291);
			this->numServicesLabel->Name = L"numServicesLabel";
			this->numServicesLabel->Size = System::Drawing::Size(204, 30);
			this->numServicesLabel->TabIndex = 2;
			this->numServicesLabel->Text = L"Number of Services";
			// 
			// serviceRadiusNum
			// 
			this->serviceRadiusNum->DecimalPlaces = 2;
			this->serviceRadiusNum->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->serviceRadiusNum->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
			this->serviceRadiusNum->Location = System::Drawing::Point(64, 332);
			this->serviceRadiusNum->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1000, 0, 0, 0 });
			this->serviceRadiusNum->Name = L"serviceRadiusNum";
			this->serviceRadiusNum->Size = System::Drawing::Size(101, 34);
			this->serviceRadiusNum->TabIndex = 3;
			this->serviceRadiusNum->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 125, 0, 0, 0 });
			// 
			// serviceCoverageRadiusLabel
			// 
			this->serviceCoverageRadiusLabel->AutoSize = true;
			this->serviceCoverageRadiusLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->serviceCoverageRadiusLabel->Location = System::Drawing::Point(175, 334);
			this->serviceCoverageRadiusLabel->Name = L"serviceCoverageRadiusLabel";
			this->serviceCoverageRadiusLabel->Size = System::Drawing::Size(176, 30);
			this->serviceCoverageRadiusLabel->TabIndex = 4;
			this->serviceCoverageRadiusLabel->Text = L"Coverage Radius";
			// 
			// customDistributionCB
			// 
			this->customDistributionCB->AutoSize = true;
			this->customDistributionCB->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->customDistributionCB->Location = System::Drawing::Point(67, 372);
			this->customDistributionCB->Name = L"customDistributionCB";
			this->customDistributionCB->Size = System::Drawing::Size(228, 34);
			this->customDistributionCB->TabIndex = 8;
			this->customDistributionCB->Text = L"Custom distribution";
			this->customDistributionCB->UseVisualStyleBackColor = true;
			this->customDistributionCB->CheckedChanged += gcnew System::EventHandler(this, &MainForm::customDistributionCB_CheckedChanged);
			// 
			// customDistributionTB
			// 
			this->customDistributionTB->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->customDistributionTB->Enabled = false;
			this->customDistributionTB->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->customDistributionTB->Location = System::Drawing::Point(301, 372);
			this->customDistributionTB->MaxLength = 60;
			this->customDistributionTB->Name = L"customDistributionTB";
			this->customDistributionTB->Size = System::Drawing::Size(281, 34);
			this->customDistributionTB->TabIndex = 9;
			// 
			// storeBoundary
			// 
			this->storeBoundary->AutoSize = true;
			this->storeBoundary->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->storeBoundary->Location = System::Drawing::Point(67, 196);
			this->storeBoundary->Name = L"storeBoundary";
			this->storeBoundary->Size = System::Drawing::Size(330, 34);
			this->storeBoundary->TabIndex = 12;
			this->storeBoundary->Text = L"Store boundary as binary map";
			this->storeBoundary->UseVisualStyleBackColor = true;
			// 
			// storeIterations
			// 
			this->storeIterations->AutoSize = true;
			this->storeIterations->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->storeIterations->Location = System::Drawing::Point(67, 222);
			this->storeIterations->Name = L"storeIterations";
			this->storeIterations->Size = System::Drawing::Size(311, 34);
			this->storeIterations->TabIndex = 13;
			this->storeIterations->Text = L"Store coverage per iteration";
			this->storeIterations->UseVisualStyleBackColor = true;
			// 
			// inputLabel
			// 
			this->inputLabel->AutoSize = true;
			this->inputLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 11, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->inputLabel->Location = System::Drawing::Point(45, 90);
			this->inputLabel->Name = L"inputLabel";
			this->inputLabel->Size = System::Drawing::Size(66, 30);
			this->inputLabel->TabIndex = 18;
			this->inputLabel->Text = L"Input";
			// 
			// maximumCoverageCB
			// 
			this->maximumCoverageCB->AutoSize = true;
			this->maximumCoverageCB->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->maximumCoverageCB->Location = System::Drawing::Point(67, 534);
			this->maximumCoverageCB->Name = L"maximumCoverageCB";
			this->maximumCoverageCB->Size = System::Drawing::Size(277, 34);
			this->maximumCoverageCB->TabIndex = 24;
			this->maximumCoverageCB->Text = L"Find maximum coverage";
			this->maximumCoverageCB->UseVisualStyleBackColor = true;
			// 
			// exactExpansionCB
			// 
			this->exactExpansionCB->AutoSize = true;
			this->exactExpansionCB->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->exactExpansionCB->Location = System::Drawing::Point(67, 464);
			this->exactExpansionCB->Name = L"exactExpansionCB";
			this->exactExpansionCB->Size = System::Drawing::Size(190, 34);
			this->exactExpansionCB->TabIndex = 26;
			this->exactExpansionCB->Text = L"Exact Expansion";
			this->exactExpansionCB->UseVisualStyleBackColor = true;
			// 
			// euclideanExpansionCB
			// 
			this->euclideanExpansionCB->AutoSize = true;
			this->euclideanExpansionCB->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->euclideanExpansionCB->Location = System::Drawing::Point(67, 437);
			this->euclideanExpansionCB->Name = L"euclideanExpansionCB";
			this->euclideanExpansionCB->Size = System::Drawing::Size(232, 34);
			this->euclideanExpansionCB->TabIndex = 25;
			this->euclideanExpansionCB->Text = L"Euclidean Expansion";
			this->euclideanExpansionCB->UseVisualStyleBackColor = true;
			// 
			// serviceConfigLabel
			// 
			this->serviceConfigLabel->AutoSize = true;
			this->serviceConfigLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 11, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->serviceConfigLabel->Location = System::Drawing::Point(45, 259);
			this->serviceConfigLabel->Name = L"serviceConfigLabel";
			this->serviceConfigLabel->Size = System::Drawing::Size(230, 30);
			this->serviceConfigLabel->TabIndex = 27;
			this->serviceConfigLabel->Text = L"Service Configuration";
			// 
			// outputLabel
			// 
			this->outputLabel->AutoSize = true;
			this->outputLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 11, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->outputLabel->Location = System::Drawing::Point(45, 163);
			this->outputLabel->Name = L"outputLabel";
			this->outputLabel->Size = System::Drawing::Size(91, 30);
			this->outputLabel->TabIndex = 28;
			this->outputLabel->Text = L"Output ";
			// 
			// algorithmConfigLabel
			// 
			this->algorithmConfigLabel->AutoSize = true;
			this->algorithmConfigLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 11, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->algorithmConfigLabel->Location = System::Drawing::Point(45, 409);
			this->algorithmConfigLabel->Name = L"algorithmConfigLabel";
			this->algorithmConfigLabel->Size = System::Drawing::Size(259, 30);
			this->algorithmConfigLabel->TabIndex = 29;
			this->algorithmConfigLabel->Text = L"Algorithm Configuration";
			// 
			// metaheuristicsLabel
			// 
			this->metaheuristicsLabel->AutoSize = true;
			this->metaheuristicsLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 11, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->metaheuristicsLabel->Location = System::Drawing::Point(45, 501);
			this->metaheuristicsLabel->Name = L"metaheuristicsLabel";
			this->metaheuristicsLabel->Size = System::Drawing::Size(164, 30);
			this->metaheuristicsLabel->TabIndex = 30;
			this->metaheuristicsLabel->Text = L"Metaheuristics ";
			// 
			// solutionDataLabel
			// 
			this->solutionDataLabel->AutoSize = true;
			this->solutionDataLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 11, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->solutionDataLabel->Location = System::Drawing::Point(45, 571);
			this->solutionDataLabel->Name = L"solutionDataLabel";
			this->solutionDataLabel->Size = System::Drawing::Size(149, 30);
			this->solutionDataLabel->TabIndex = 31;
			this->solutionDataLabel->Text = L"Solution Data";
			// 
			// solutionDataRTB
			// 
			this->solutionDataRTB->Enabled = false;
			this->solutionDataRTB->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->solutionDataRTB->Location = System::Drawing::Point(67, 604);
			this->solutionDataRTB->Name = L"solutionDataRTB";
			this->solutionDataRTB->Size = System::Drawing::Size(515, 145);
			this->solutionDataRTB->TabIndex = 32;
			this->solutionDataRTB->Text = L"";
			// 
			// openImageBT
			// 
			this->openImageBT->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10));
			this->openImageBT->Location = System::Drawing::Point(464, 123);
			this->openImageBT->Name = L"openImageBT";
			this->openImageBT->Size = System::Drawing::Size(113, 26);
			this->openImageBT->TabIndex = 20;
			this->openImageBT->Text = L"Open File";
			this->openImageBT->UseVisualStyleBackColor = true;
			this->openImageBT->Click += gcnew System::EventHandler(this, &MainForm::openImageBT_Click);
			// 
			// MainForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(18, 45);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(644, 807);
			this->Controls->Add(this->openImageBT);
			this->Controls->Add(this->solutionDataRTB);
			this->Controls->Add(this->inputPathTB);
			this->Controls->Add(this->solutionDataLabel);
			this->Controls->Add(this->metaheuristicsLabel);
			this->Controls->Add(this->algorithmConfigLabel);
			this->Controls->Add(this->outputLabel);
			this->Controls->Add(this->serviceConfigLabel);
			this->Controls->Add(this->exactExpansionCB);
			this->Controls->Add(this->euclideanExpansionCB);
			this->Controls->Add(this->maximumCoverageCB);
			this->Controls->Add(this->inputLabel);
			this->Controls->Add(this->storeIterations);
			this->Controls->Add(this->storeBoundary);
			this->Controls->Add(this->customDistributionTB);
			this->Controls->Add(this->customDistributionCB);
			this->Controls->Add(this->tableLayoutPanel1);
			this->Controls->Add(this->serviceCoverageRadiusLabel);
			this->Controls->Add(this->serviceRadiusNum);
			this->Controls->Add(this->numServicesLabel);
			this->Controls->Add(this->titleLabel);
			this->Controls->Add(this->numberOfServicesNum);
			this->Font = (gcnew System::Drawing::Font(L"Segoe UI", 16, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedToolWindow;
			this->Margin = System::Windows::Forms::Padding(6, 7, 6, 7);
			this->Name = L"MainForm";
			this->SizeGripStyle = System::Windows::Forms::SizeGripStyle::Show;
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterScreen;
			this->tableLayoutPanel1->ResumeLayout(false);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numberOfServicesNum))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->serviceRadiusNum))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
		private: int iterations;
		
		private: System::Void openImageBT_Click(System::Object^ sender, System::EventArgs^ e) {
			OpenFileDialog^ openFileDialog1 = gcnew OpenFileDialog();

			openFileDialog1->Title = "Select an Image";

			if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
				String^ filePath = openFileDialog1->FileName;

				inputPathTB->Text = filePath;
			}
		}

	   private: System::String^ IncrementImageNumber(System::String^ input, int newNumber)
	   {
		   // Convert the input string to a std::string
		   std::string inputStr = msclr::interop::marshal_as<std::string>(input);

		   // Find the last sequence of digits in the string
		   size_t pos = inputStr.find_last_of("0123456789");

		   // If a sequence of digits was found, replace it with the new number
		   if (pos != std::string::npos)
		   {
			   inputStr.replace(pos, std::string::npos, std::to_string(newNumber));
		   }

		   // Convert the result back to a System::String^ and return it
		   return msclr::interop::marshal_as<System::String^>(inputStr);
	   }

	private: System::Void runButton_Click(System::Object^ sender, System::EventArgs^ e) {
		OpenFileDialog^ openFileDialog1 = gcnew OpenFileDialog();

		openFileDialog1->Title = "Select an Image";

		if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			config currentConfig;
			iterations++;

			String^ filePath = openFileDialog1->FileName;

			currentConfig.domainPath = marshal_as<std::string>(filePath);
			currentConfig.outputFileName = "output";//marshal_as<std::string>(outputImageTextBox->Text);

			currentConfig.numSources = static_cast<int>(numberOfServicesNum->Value);
			currentConfig.radius = static_cast<float>(serviceRadiusNum->Value);

			currentConfig.randomSources = !(customDistributionCB->Checked);
		//	std::cout << currentConfig.randomSources << " " << std::endl;

			/*
			if (customDistributionCB->Checked) {
				System::String^ inputString = customDistributionTB->Text;

				std::string inputStdString = marshal_as<std::string>(inputString);
				std::stringstream ss(inputStdString);
				std::vector<int> intVector;

				while (ss.good()) {
					std::string substr;
					getline(ss, substr, ',');
					int value = std::stoi(substr);
					intVector.push_back(value);
				}

				currentConfig.sources = new int[intVector.size()];
				currentConfig.numSources = intVector.size() / 2;
				std::copy(intVector.begin(), intVector.end(), currentConfig.sources);
			}
			*/

			currentConfig.verboseMode = false;
			currentConfig.storeBoundary = storeBoundary->Checked;
			currentConfig.storeIterationContent = storeIterations -> Checked;
			currentConfig.storeFinalResult = false;//storeFinalResult->Checked;

			runExactExpansion(currentConfig);

			std::string fileOutputPath = "output/" + currentConfig.outputFileName + ".png";
			System::String^ filePathStr = msclr::interop::marshal_as<System::String^>(fileOutputPath);

			Bitmap^ image = gcnew Bitmap(filePathStr);
			ImageDisplay^ form = gcnew ImageDisplay(image);
			form->Show();

		//	System::String^ currentText = outputImageTextBox->Text;
		//	System::String^ newText = IncrementImageNumber(currentText, iterations);

			//outputImageTextBox->Text = newText;
		}
	}
	
	private: System::Void customDistributionCB_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
		customDistributionTB->Enabled = customDistributionCB->Checked;
	
		if (!customDistributionTB->Enabled) {
			customDistributionTB->Text = "x1,y1,x2,y2,x3,y3...";
		}
		else
			customDistributionTB->Text = "";
	}

	private: System::Void exitButton_Click(System::Object^ sender, System::EventArgs^ e) {
		Application::Exit();
	}


};
}
