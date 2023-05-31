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





	private: System::Windows::Forms::Label^ solutionDataLabel;
	private: System::Windows::Forms::RichTextBox^ solutionDataRTB;
	private: System::Windows::Forms::Button^ openImageBT;

	private: System::Windows::Forms::TextBox^ inputPathTB;


	private: System::Windows::Forms::Label^ label2;
	private: System::Windows::Forms::NumericUpDown^ numGenerations;

	private: System::Windows::Forms::Label^ label3;
	private: System::Windows::Forms::NumericUpDown^ populationSize;

	private: System::Windows::Forms::Label^ label4;
	private: System::Windows::Forms::NumericUpDown^ mutationRate;

	private: System::Windows::Forms::Label^ label5;
	private: System::Windows::Forms::NumericUpDown^ stopThreshold;































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
			this->solutionDataLabel = (gcnew System::Windows::Forms::Label());
			this->solutionDataRTB = (gcnew System::Windows::Forms::RichTextBox());
			this->openImageBT = (gcnew System::Windows::Forms::Button());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->numGenerations = (gcnew System::Windows::Forms::NumericUpDown());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->populationSize = (gcnew System::Windows::Forms::NumericUpDown());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->mutationRate = (gcnew System::Windows::Forms::NumericUpDown());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->stopThreshold = (gcnew System::Windows::Forms::NumericUpDown());
			this->tableLayoutPanel1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numberOfServicesNum))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->serviceRadiusNum))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numGenerations))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->populationSize))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->mutationRate))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->stopThreshold))->BeginInit();
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
			this->inputPathTB->Size = System::Drawing::Size(420, 34);
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
			this->tableLayoutPanel1->Location = System::Drawing::Point(12, 866);
			this->tableLayoutPanel1->Name = L"tableLayoutPanel1";
			this->tableLayoutPanel1->RowCount = 1;
			this->tableLayoutPanel1->RowStyles->Add((gcnew System::Windows::Forms::RowStyle(System::Windows::Forms::SizeType::Percent, 50)));
			this->tableLayoutPanel1->Size = System::Drawing::Size(699, 42);
			this->tableLayoutPanel1->TabIndex = 7;
			// 
			// exitButton
			// 
			this->exitButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left)
				| System::Windows::Forms::AnchorStyles::Right));
			this->exitButton->Location = System::Drawing::Point(3, 3);
			this->exitButton->Name = L"exitButton";
			this->exitButton->Size = System::Drawing::Size(343, 36);
			this->exitButton->TabIndex = 0;
			this->exitButton->Text = L"Exit";
			this->exitButton->UseVisualStyleBackColor = true;
			this->exitButton->Click += gcnew System::EventHandler(this, &MainForm::exitButton_Click);
			// 
			// runButton
			// 
			this->runButton->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left)
				| System::Windows::Forms::AnchorStyles::Right));
			this->runButton->Location = System::Drawing::Point(352, 3);
			this->runButton->Name = L"runButton";
			this->runButton->Size = System::Drawing::Size(344, 36);
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
			this->numberOfServicesNum->Location = System::Drawing::Point(67, 175);
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
			this->numServicesLabel->Location = System::Drawing::Point(175, 175);
			this->numServicesLabel->Name = L"numServicesLabel";
			this->numServicesLabel->Size = System::Drawing::Size(92, 30);
			this->numServicesLabel->TabIndex = 2;
			this->numServicesLabel->Text = L"Services";
			// 
			// serviceRadiusNum
			// 
			this->serviceRadiusNum->DecimalPlaces = 2;
			this->serviceRadiusNum->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->serviceRadiusNum->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
			this->serviceRadiusNum->Location = System::Drawing::Point(275, 175);
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
			this->serviceCoverageRadiusLabel->Location = System::Drawing::Point(382, 175);
			this->serviceCoverageRadiusLabel->Name = L"serviceCoverageRadiusLabel";
			this->serviceCoverageRadiusLabel->Size = System::Drawing::Size(74, 30);
			this->serviceCoverageRadiusLabel->TabIndex = 4;
			this->serviceCoverageRadiusLabel->Text = L"Range";
			// 
			// customDistributionCB
			// 
			this->customDistributionCB->AutoSize = true;
			this->customDistributionCB->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->customDistributionCB->Location = System::Drawing::Point(67, 224);
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
			this->customDistributionTB->Location = System::Drawing::Point(311, 224);
			this->customDistributionTB->MaxLength = 60;
			this->customDistributionTB->Name = L"customDistributionTB";
			this->customDistributionTB->Size = System::Drawing::Size(342, 34);
			this->customDistributionTB->TabIndex = 9;
			// 
			// storeBoundary
			// 
			this->storeBoundary->AutoSize = true;
			this->storeBoundary->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->storeBoundary->Location = System::Drawing::Point(67, 304);
			this->storeBoundary->Name = L"storeBoundary";
			this->storeBoundary->Size = System::Drawing::Size(189, 34);
			this->storeBoundary->TabIndex = 12;
			this->storeBoundary->Text = L"Store boundary";
			this->storeBoundary->UseVisualStyleBackColor = true;
			// 
			// storeIterations
			// 
			this->storeIterations->AutoSize = true;
			this->storeIterations->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->storeIterations->Location = System::Drawing::Point(311, 304);
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
			this->inputLabel->Size = System::Drawing::Size(252, 30);
			this->inputLabel->TabIndex = 18;
			this->inputLabel->Text = L"Coverage Configuration";
			// 
			// maximumCoverageCB
			// 
			this->maximumCoverageCB->AutoSize = true;
			this->maximumCoverageCB->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->maximumCoverageCB->Location = System::Drawing::Point(69, 505);
			this->maximumCoverageCB->Name = L"maximumCoverageCB";
			this->maximumCoverageCB->Size = System::Drawing::Size(159, 34);
			this->maximumCoverageCB->TabIndex = 24;
			this->maximumCoverageCB->Text = L"MCLP Mode";
			this->maximumCoverageCB->UseVisualStyleBackColor = true;
			// 
			// exactExpansionCB
			// 
			this->exactExpansionCB->AutoSize = true;
			this->exactExpansionCB->Checked = true;
			this->exactExpansionCB->CheckState = System::Windows::Forms::CheckState::Checked;
			this->exactExpansionCB->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->exactExpansionCB->Location = System::Drawing::Point(311, 264);
			this->exactExpansionCB->Name = L"exactExpansionCB";
			this->exactExpansionCB->Size = System::Drawing::Size(190, 34);
			this->exactExpansionCB->TabIndex = 26;
			this->exactExpansionCB->Text = L"Exact Expansion";
			this->exactExpansionCB->UseVisualStyleBackColor = true;
			this->exactExpansionCB->CheckedChanged += gcnew System::EventHandler(this, &MainForm::exactExpansionCB_CheckedChanged);
			// 
			// euclideanExpansionCB
			// 
			this->euclideanExpansionCB->AutoSize = true;
			this->euclideanExpansionCB->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->euclideanExpansionCB->Location = System::Drawing::Point(67, 264);
			this->euclideanExpansionCB->Name = L"euclideanExpansionCB";
			this->euclideanExpansionCB->Size = System::Drawing::Size(232, 34);
			this->euclideanExpansionCB->TabIndex = 25;
			this->euclideanExpansionCB->Text = L"Euclidean Expansion";
			this->euclideanExpansionCB->UseVisualStyleBackColor = true;
			this->euclideanExpansionCB->CheckedChanged += gcnew System::EventHandler(this, &MainForm::euclideanExpansionCB_CheckedChanged);
			// 
			// serviceConfigLabel
			// 
			this->serviceConfigLabel->AutoSize = true;
			this->serviceConfigLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 11, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->serviceConfigLabel->Location = System::Drawing::Point(45, 376);
			this->serviceConfigLabel->Name = L"serviceConfigLabel";
			this->serviceConfigLabel->Size = System::Drawing::Size(284, 30);
			this->serviceConfigLabel->TabIndex = 27;
			this->serviceConfigLabel->Text = L"MCLP Solver Configuration";
			// 
			// solutionDataLabel
			// 
			this->solutionDataLabel->AutoSize = true;
			this->solutionDataLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI Semibold", 11, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->solutionDataLabel->Location = System::Drawing::Point(45, 557);
			this->solutionDataLabel->Name = L"solutionDataLabel";
			this->solutionDataLabel->Size = System::Drawing::Size(149, 30);
			this->solutionDataLabel->TabIndex = 31;
			this->solutionDataLabel->Text = L"Solution Data";
			// 
			// solutionDataRTB
			// 
			this->solutionDataRTB->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->solutionDataRTB->Location = System::Drawing::Point(67, 604);
			this->solutionDataRTB->Name = L"solutionDataRTB";
			this->solutionDataRTB->Size = System::Drawing::Size(586, 256);
			this->solutionDataRTB->TabIndex = 32;
			this->solutionDataRTB->Text = L"";
			// 
			// openImageBT
			// 
			this->openImageBT->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10));
			this->openImageBT->Location = System::Drawing::Point(498, 123);
			this->openImageBT->Name = L"openImageBT";
			this->openImageBT->Size = System::Drawing::Size(155, 26);
			this->openImageBT->TabIndex = 20;
			this->openImageBT->Text = L"Open Domain";
			this->openImageBT->UseVisualStyleBackColor = true;
			this->openImageBT->Click += gcnew System::EventHandler(this, &MainForm::openImageBT_Click);
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->label2->Location = System::Drawing::Point(175, 423);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(129, 30);
			this->label2->TabIndex = 34;
			this->label2->Text = L"Generations";
			// 
			// numGenerations
			// 
			this->numGenerations->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->numGenerations->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
			this->numGenerations->Location = System::Drawing::Point(67, 423);
			this->numGenerations->Name = L"numGenerations";
			this->numGenerations->Size = System::Drawing::Size(101, 34);
			this->numGenerations->TabIndex = 33;
			this->numGenerations->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->label3->Location = System::Drawing::Point(175, 463);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(160, 30);
			this->label3->TabIndex = 38;
			this->label3->Text = L"Population Size";
			// 
			// populationSize
			// 
			this->populationSize->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->populationSize->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
			this->populationSize->Location = System::Drawing::Point(67, 463);
			this->populationSize->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 500, 0, 0, 0 });
			this->populationSize->Name = L"populationSize";
			this->populationSize->Size = System::Drawing::Size(101, 34);
			this->populationSize->TabIndex = 37;
			this->populationSize->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 50, 0, 0, 0 });
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->label4->Location = System::Drawing::Point(468, 423);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(149, 30);
			this->label4->TabIndex = 40;
			this->label4->Text = L"Mutation Rate";
			// 
			// mutationRate
			// 
			this->mutationRate->DecimalPlaces = 2;
			this->mutationRate->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->mutationRate->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 131072 });
			this->mutationRate->Location = System::Drawing::Point(361, 423);
			this->mutationRate->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->mutationRate->Name = L"mutationRate";
			this->mutationRate->Size = System::Drawing::Size(101, 34);
			this->mutationRate->TabIndex = 39;
			this->mutationRate->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 20, 0, 0, 131072 });
			// 
			// label5
			// 
			this->label5->AutoSize = true;
			this->label5->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11));
			this->label5->Location = System::Drawing::Point(468, 463);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(187, 30);
			this->label5->TabIndex = 42;
			this->label5->Text = L"Desired Coverage";
			// 
			// stopThreshold
			// 
			this->stopThreshold->DecimalPlaces = 2;
			this->stopThreshold->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->stopThreshold->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
			this->stopThreshold->Location = System::Drawing::Point(361, 463);
			this->stopThreshold->Name = L"stopThreshold";
			this->stopThreshold->Size = System::Drawing::Size(101, 34);
			this->stopThreshold->TabIndex = 41;
			this->stopThreshold->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 90, 0, 0, 0 });
			// 
			// MainForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(18, 45);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(723, 918);
			this->Controls->Add(this->label5);
			this->Controls->Add(this->stopThreshold);
			this->Controls->Add(this->label4);
			this->Controls->Add(this->mutationRate);
			this->Controls->Add(this->label3);
			this->Controls->Add(this->populationSize);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->numGenerations);
			this->Controls->Add(this->openImageBT);
			this->Controls->Add(this->solutionDataRTB);
			this->Controls->Add(this->inputPathTB);
			this->Controls->Add(this->solutionDataLabel);
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
			this->Load += gcnew System::EventHandler(this, &MainForm::MainForm_Load);
			this->tableLayoutPanel1->ResumeLayout(false);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numberOfServicesNum))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->serviceRadiusNum))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numGenerations))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->populationSize))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->mutationRate))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->stopThreshold))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	
	private: System::Void openImageBT_Click(System::Object^ sender, System::EventArgs^ e) {
		OpenFileDialog^ openFileDialog1 = gcnew OpenFileDialog();

		openFileDialog1->Title = "Select an Image";

		if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			String^ filePath = openFileDialog1->FileName;

			inputPathTB->Text = filePath;
		}
	}

	private: System::Void runButton_Click(System::Object^ sender, System::EventArgs^ e) {
		SystemParameters config;
		AlgorithmParameters algParams;
		OptimizationParameters optParams;

		config.imagePath = marshal_as<std::string>(inputPathTB->Text);

		std::size_t lastSlash = config.imagePath.find_last_of("\\/");
		std::string imageName = config.imagePath.substr(lastSlash + 1);
		auto currentTime = std::chrono::system_clock::now();
		std::string timestampStr = std::to_string(std::chrono::system_clock::to_time_t(currentTime));

		config.imageName = timestampStr + "_" + imageName;

		config.storeBoundary = storeBoundary->Checked;
		config.storeIterCoverage = storeIterations->Checked;

		algParams.numServices = static_cast<int>(numberOfServicesNum->Value);
		algParams.serviceRadius = static_cast<float>(serviceRadiusNum->Value);
	
		config.customDistribution = customDistributionCB->Checked;
		config.serviceDistribution = marshal_as<std::string>(customDistributionTB->Text);

		algParams.useEuclideanExpansion = euclideanExpansionCB->Checked;
		algParams.useExactExpansion = exactExpansionCB->Checked;

		config.maxCoverage = maximumCoverageCB->Checked;

		optParams.numGenerations = static_cast<int>(numGenerations->Value);
		optParams.populationSize = static_cast<int>(populationSize->Value);
		optParams.mutationRate = static_cast<float>(mutationRate->Value);
		optParams.stopThreshold = static_cast<float>(stopThreshold->Value);

		if (algParams.useExactExpansion)
			runExactExpansion(config, algParams, optParams);
		else
			runEuclideanExpansion(config);

		solutionDataRTB->Text = marshal_as<String^>(config.solutionData);

		std::string fileOutputPath = "output/" + config.imageName;

		System::String^ filePathStr = msclr::interop::marshal_as<System::String^>(fileOutputPath);

		Bitmap^ image = gcnew Bitmap(filePathStr);
		ImageDisplay^ form = gcnew ImageDisplay(image);
		form->Show();
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

	private: System::Void euclideanExpansionCB_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
		if (euclideanExpansionCB->Checked) {
			exactExpansionCB->Checked = false;
		}
	}

	private: System::Void exactExpansionCB_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
		if (exactExpansionCB->Checked) {
			euclideanExpansionCB->Checked = false;
		}
	}
private: System::Void MainForm_Load(System::Object^ sender, System::EventArgs^ e) {
}
};
}
