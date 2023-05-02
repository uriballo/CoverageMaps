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
	private: System::Windows::Forms::NumericUpDown^ numericUpDown1;
	protected:
	private: System::Windows::Forms::Label^ titleLabel;
	private: System::Windows::Forms::Label^ numSourcesLabel;
	private: System::Windows::Forms::Label^ radiusLabel;
	private: System::Windows::Forms::NumericUpDown^ numericUpDown2;
	private: System::Windows::Forms::TextBox^ outputImageTextBox;

	private: System::Windows::Forms::Label^ outputName;
	private: System::Windows::Forms::TableLayoutPanel^ tableLayoutPanel1;
	private: System::Windows::Forms::Button^ exitButton;
	private: System::Windows::Forms::Button^ runButton;
	private: System::Windows::Forms::OpenFileDialog^ openFileDialog1;
	private: System::Windows::Forms::CheckBox^ customDistributionCB;
	private: System::Windows::Forms::TextBox^ customDistributionTB;




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
			this->numericUpDown1 = (gcnew System::Windows::Forms::NumericUpDown());
			this->titleLabel = (gcnew System::Windows::Forms::Label());
			this->numSourcesLabel = (gcnew System::Windows::Forms::Label());
			this->radiusLabel = (gcnew System::Windows::Forms::Label());
			this->numericUpDown2 = (gcnew System::Windows::Forms::NumericUpDown());
			this->outputImageTextBox = (gcnew System::Windows::Forms::TextBox());
			this->outputName = (gcnew System::Windows::Forms::Label());
			this->tableLayoutPanel1 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->exitButton = (gcnew System::Windows::Forms::Button());
			this->runButton = (gcnew System::Windows::Forms::Button());
			this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->customDistributionCB = (gcnew System::Windows::Forms::CheckBox());
			this->customDistributionTB = (gcnew System::Windows::Forms::TextBox());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown1))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown2))->BeginInit();
			this->tableLayoutPanel1->SuspendLayout();
			this->SuspendLayout();
			// 
			// numericUpDown1
			// 
			this->numericUpDown1->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->numericUpDown1->Location = System::Drawing::Point(50, 160);
			this->numericUpDown1->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
			this->numericUpDown1->Name = L"numericUpDown1";
			this->numericUpDown1->Size = System::Drawing::Size(207, 34);
			this->numericUpDown1->TabIndex = 0;
			this->numericUpDown1->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 2, 0, 0, 0 });
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
			// numSourcesLabel
			// 
			this->numSourcesLabel->AutoSize = true;
			this->numSourcesLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->numSourcesLabel->Location = System::Drawing::Point(44, 115);
			this->numSourcesLabel->Name = L"numSourcesLabel";
			this->numSourcesLabel->Size = System::Drawing::Size(218, 32);
			this->numSourcesLabel->TabIndex = 2;
			this->numSourcesLabel->Text = L"Number of sources";
			// 
			// radiusLabel
			// 
			this->radiusLabel->AutoSize = true;
			this->radiusLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->radiusLabel->Location = System::Drawing::Point(512, 115);
			this->radiusLabel->Name = L"radiusLabel";
			this->radiusLabel->Size = System::Drawing::Size(84, 32);
			this->radiusLabel->TabIndex = 4;
			this->radiusLabel->Text = L"Radius";
			// 
			// numericUpDown2
			// 
			this->numericUpDown2->DecimalPlaces = 2;
			this->numericUpDown2->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->numericUpDown2->Location = System::Drawing::Point(518, 160);
			this->numericUpDown2->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1000, 0, 0, 0 });
			this->numericUpDown2->Name = L"numericUpDown2";
			this->numericUpDown2->Size = System::Drawing::Size(207, 34);
			this->numericUpDown2->TabIndex = 3;
			this->numericUpDown2->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100, 0, 0, 0 });
			// 
			// outputImageTextBox
			// 
			this->outputImageTextBox->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->outputImageTextBox->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->outputImageTextBox->Location = System::Drawing::Point(331, 304);
			this->outputImageTextBox->MaxLength = 12;
			this->outputImageTextBox->Name = L"outputImageTextBox";
			this->outputImageTextBox->Size = System::Drawing::Size(394, 34);
			this->outputImageTextBox->TabIndex = 5;
			this->outputImageTextBox->Text = L"output1";
			// 
			// outputName
			// 
			this->outputName->AutoSize = true;
			this->outputName->Font = (gcnew System::Drawing::Font(L"Segoe UI", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->outputName->Location = System::Drawing::Point(44, 303);
			this->outputName->Name = L"outputName";
			this->outputName->Size = System::Drawing::Size(230, 32);
			this->outputName->TabIndex = 6;
			this->outputName->Text = L"Output image name";
			this->outputName->Click += gcnew System::EventHandler(this, &MainForm::label1_Click);
			// 
			// tableLayoutPanel1
			// 
			this->tableLayoutPanel1->ColumnCount = 2;
			this->tableLayoutPanel1->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
				50)));
			this->tableLayoutPanel1->ColumnStyles->Add((gcnew System::Windows::Forms::ColumnStyle(System::Windows::Forms::SizeType::Percent,
				50)));
			this->tableLayoutPanel1->Controls->Add(this->exitButton, 0, 0);
			this->tableLayoutPanel1->Controls->Add(this->runButton, 1, 0);
			this->tableLayoutPanel1->Location = System::Drawing::Point(13, 663);
			this->tableLayoutPanel1->Name = L"tableLayoutPanel1";
			this->tableLayoutPanel1->RowCount = 1;
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 50)));
			this->tableLayoutPanel1->Size = System::Drawing::Size(753, 69);
			this->tableLayoutPanel1->TabIndex = 7;
			// 
			// exitButton
			// 
			this->exitButton->Location = System::Drawing::Point(3, 3);
			this->exitButton->Name = L"exitButton";
			this->exitButton->Size = System::Drawing::Size(370, 63);
			this->exitButton->TabIndex = 0;
			this->exitButton->Text = L"EXIT";
			this->exitButton->UseVisualStyleBackColor = true;
			this->exitButton->Click += gcnew System::EventHandler(this, &MainForm::exitButton_Click);
			// 
			// runButton
			// 
			this->runButton->Location = System::Drawing::Point(379, 3);
			this->runButton->Name = L"runButton";
			this->runButton->Size = System::Drawing::Size(371, 63);
			this->runButton->TabIndex = 1;
			this->runButton->Text = L"RUN";
			this->runButton->UseVisualStyleBackColor = true;
			this->runButton->Click += gcnew System::EventHandler(this, &MainForm::runButton_Click);
			// 
			// openFileDialog1
			// 
			this->openFileDialog1->FileName = L"openFileDialog1";
			// 
			// customDistributionCB
			// 
			this->customDistributionCB->AutoSize = true;
			this->customDistributionCB->Font = (gcnew System::Drawing::Font(L"Segoe UI", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->customDistributionCB->Location = System::Drawing::Point(50, 398);
			this->customDistributionCB->Name = L"customDistributionCB";
			this->customDistributionCB->Size = System::Drawing::Size(251, 36);
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
			this->customDistributionTB->Location = System::Drawing::Point(331, 400);
			this->customDistributionTB->MaxLength = 60;
			this->customDistributionTB->Name = L"customDistributionTB";
			this->customDistributionTB->Size = System::Drawing::Size(394, 34);
			this->customDistributionTB->TabIndex = 9;
			// 
			// MainForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(18, 45);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(778, 744);
			this->Controls->Add(this->customDistributionTB);
			this->Controls->Add(this->customDistributionCB);
			this->Controls->Add(this->tableLayoutPanel1);
			this->Controls->Add(this->outputName);
			this->Controls->Add(this->outputImageTextBox);
			this->Controls->Add(this->radiusLabel);
			this->Controls->Add(this->numericUpDown2);
			this->Controls->Add(this->numSourcesLabel);
			this->Controls->Add(this->titleLabel);
			this->Controls->Add(this->numericUpDown1);
			this->Font = (gcnew System::Drawing::Font(L"Segoe UI", 16, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedToolWindow;
			this->Margin = System::Windows::Forms::Padding(6, 7, 6, 7);
			this->Name = L"MainForm";
			this->ShowIcon = false;
			this->SizeGripStyle = System::Windows::Forms::SizeGripStyle::Show;
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterScreen;
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown1))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown2))->EndInit();
			this->tableLayoutPanel1->ResumeLayout(false);
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
		private: int iterations;
	private: System::Void label1_Click(System::Object^ sender, System::EventArgs^ e) {
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
			currentConfig.outputFileName = marshal_as<std::string>(outputImageTextBox->Text);

			currentConfig.numSources = static_cast<int>(numericUpDown1->Value);
			currentConfig.radius = static_cast<float>(numericUpDown2->Value);

			currentConfig.randomSources = true;//!(customDistributionCB->Checked);

			if (!currentConfig.randomSources) {
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
				std::copy(intVector.begin(), intVector.end(), currentConfig.sources);
			}

			runExactExpansion(currentConfig);

			std::string fileOutputPath = "output/" + currentConfig.outputFileName + ".png";
			System::String^ filePathStr = msclr::interop::marshal_as<System::String^>(fileOutputPath);

			Bitmap^ image = gcnew Bitmap(filePathStr);
			ImageDisplay^ form = gcnew ImageDisplay(image);
			form->Show();

			System::String^ currentText = outputImageTextBox->Text;
			System::String^ newText = IncrementImageNumber(currentText, iterations);

			outputImageTextBox->Text = newText;
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
