#pragma once
#include "EuclideanExpansion.h"
#include "FeedbackExpansion.h"
#include "ImageDisplay.h"
#include <msclr/marshal_cppstd.h>

using namespace msclr::interop;

namespace CUDACoverageMaps {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for GUI
	/// </summary>
	public ref class GUI : public System::Windows::Forms::Form
	{
	public:
		GUI(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~GUI()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Button^ button1;
	private: System::Windows::Forms::NumericUpDown^ numSourcesIN;

	private: System::Windows::Forms::Label^ label1;
	private: System::Windows::Forms::Label^ label2;
	private: System::Windows::Forms::NumericUpDown^ serviceRadiusIN;
	private: System::Windows::Forms::CheckBox^ randomDistCB;
	private: System::Windows::Forms::CheckBox^ resultsCB;
	private: System::Windows::Forms::OpenFileDialog^ openFileDialog1;
	private: System::Windows::Forms::TableLayoutPanel^ tableLayoutPanel1;
	private: System::Windows::Forms::Button^ button2;
	private: System::Windows::Forms::CheckBox^ checkBox1;







	protected:

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container^ components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->numSourcesIN = (gcnew System::Windows::Forms::NumericUpDown());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->serviceRadiusIN = (gcnew System::Windows::Forms::NumericUpDown());
			this->randomDistCB = (gcnew System::Windows::Forms::CheckBox());
			this->resultsCB = (gcnew System::Windows::Forms::CheckBox());
			this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->tableLayoutPanel1 = (gcnew System::Windows::Forms::TableLayoutPanel());
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->checkBox1 = (gcnew System::Windows::Forms::CheckBox());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numSourcesIN))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->serviceRadiusIN))->BeginInit();
			this->tableLayoutPanel1->SuspendLayout();
			this->SuspendLayout();
			// 
			// button1
			// 
			this->button1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left)
				| System::Windows::Forms::AnchorStyles::Right));
			this->button1->Location = System::Drawing::Point(263, 3);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(254, 84);
			this->button1->TabIndex = 0;
			this->button1->Text = L"RUN";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &GUI::button1_Click);
			// 
			// numSourcesIN
			// 
			this->numSourcesIN->Location = System::Drawing::Point(16, 46);
			this->numSourcesIN->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 0 });
			this->numSourcesIN->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->numSourcesIN->Name = L"numSourcesIN";
			this->numSourcesIN->Size = System::Drawing::Size(193, 26);
			this->numSourcesIN->TabIndex = 1;
			this->numSourcesIN->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(12, 9);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(144, 20);
			this->label1->TabIndex = 2;
			this->label1->Text = L"Number of services";
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(275, 9);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(108, 20);
			this->label2->TabIndex = 4;
			this->label2->Text = L"Service radius";
			// 
			// serviceRadiusIN
			// 
			this->serviceRadiusIN->DecimalPlaces = 2;
			this->serviceRadiusIN->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
			this->serviceRadiusIN->Location = System::Drawing::Point(279, 46);
			this->serviceRadiusIN->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1000, 0, 0, 0 });
			this->serviceRadiusIN->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
			this->serviceRadiusIN->Name = L"serviceRadiusIN";
			this->serviceRadiusIN->Size = System::Drawing::Size(191, 26);
			this->serviceRadiusIN->TabIndex = 3;
			this->serviceRadiusIN->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100, 0, 0, 0 });
			this->serviceRadiusIN->ValueChanged += gcnew System::EventHandler(this, &GUI::numericUpDown1_ValueChanged);
			// 
			// randomDistCB
			// 
			this->randomDistCB->AutoSize = true;
			this->randomDistCB->Checked = true;
			this->randomDistCB->CheckState = System::Windows::Forms::CheckState::Checked;
			this->randomDistCB->Location = System::Drawing::Point(19, 130);
			this->randomDistCB->Name = L"randomDistCB";
			this->randomDistCB->Size = System::Drawing::Size(177, 24);
			this->randomDistCB->TabIndex = 5;
			this->randomDistCB->Text = L"Random distribution";
			this->randomDistCB->UseVisualStyleBackColor = true;
			// 
			// resultsCB
			// 
			this->resultsCB->AutoSize = true;
			this->resultsCB->Checked = true;
			this->resultsCB->CheckState = System::Windows::Forms::CheckState::Checked;
			this->resultsCB->Location = System::Drawing::Point(19, 179);
			this->resultsCB->Name = L"resultsCB";
			this->resultsCB->Size = System::Drawing::Size(126, 24);
			this->resultsCB->TabIndex = 6;
			this->resultsCB->Text = L"Show results";
			this->resultsCB->UseVisualStyleBackColor = true;
			// 
			// openFileDialog1
			// 
			this->openFileDialog1->FileName = L"openFileDialog1";
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
			this->tableLayoutPanel1->Controls->Add(this->button1, 1, 0);
			this->tableLayoutPanel1->Controls->Add(this->button2, 0, 0);
			this->tableLayoutPanel1->Location = System::Drawing::Point(16, 335);
			this->tableLayoutPanel1->Name = L"tableLayoutPanel1";
			this->tableLayoutPanel1->RowCount = 1;
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 50)));
			this->tableLayoutPanel1->Size = System::Drawing::Size(520, 90);
			this->tableLayoutPanel1->TabIndex = 7;
			// 
			// button2
			// 
			this->button2->BackColor = System::Drawing::SystemColors::ButtonHighlight;
			this->button2->Location = System::Drawing::Point(3, 3);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(254, 84);
			this->button2->TabIndex = 1;
			this->button2->Text = L"EXIT";
			this->button2->UseVisualStyleBackColor = false;
			this->button2->Click += gcnew System::EventHandler(this, &GUI::button2_Click);
			// 
			// checkBox1
			// 
			this->checkBox1->AutoSize = true;
			this->checkBox1->Location = System::Drawing::Point(19, 223);
			this->checkBox1->Name = L"checkBox1";
			this->checkBox1->Size = System::Drawing::Size(139, 24);
			this->checkBox1->TabIndex = 8;
			this->checkBox1->Text = L"Verbose mode";
			this->checkBox1->UseVisualStyleBackColor = true;
			// 
			// GUI
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(9, 20);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(548, 437);
			this->Controls->Add(this->checkBox1);
			this->Controls->Add(this->tableLayoutPanel1);
			this->Controls->Add(this->resultsCB);
			this->Controls->Add(this->randomDistCB);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->serviceRadiusIN);
			this->Controls->Add(this->label1);
			this->Controls->Add(this->numSourcesIN);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedToolWindow;
			this->Name = L"GUI";
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterScreen;
			this->Text = L"Coverage Map";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numSourcesIN))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->serviceRadiusIN))->EndInit();
			this->tableLayoutPanel1->ResumeLayout(false);
			this->ResumeLayout(false);
			this->PerformLayout();

		}

#pragma endregion

private:
	System::Void button1_Click(System::Object^ sender, System::EventArgs^ e) {
		OpenFileDialog^ openFileDialog1 = gcnew OpenFileDialog();

		openFileDialog1->Title = "Select an Image";

		if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			// Reset the state before assigning new values
			config currentConfig;

			// Get the file path from the file dialog
			String^ filePath = openFileDialog1->FileName;

			currentConfig.domainPath = marshal_as<std::string>(filePath);
			currentConfig.numSources = static_cast<int>(numSourcesIN->Value);
			currentConfig.radius = static_cast<float>(serviceRadiusIN->Value);
			currentConfig.showResults = resultsCB->Checked;
			currentConfig.randomSources = randomDistCB->Checked;
			currentConfig.displayHeatMap = false;

			// Run the simulation with the current configuration
			runExactExpansion(currentConfig);

			// Load the output image into a Bitmap and display it in a new form
			Bitmap^ image = gcnew Bitmap("output/outputImage.png");
			ImageDisplay^ form = gcnew ImageDisplay(image);
			form->Show();
		}
	}



	private: System::Void numericUpDown1_ValueChanged(System::Object^ sender, System::EventArgs^ e) {
	
	}
	
	private: System::Void button2_Click(System::Object^ sender, System::EventArgs^ e) {
		Application::Exit();
	}
};
}
