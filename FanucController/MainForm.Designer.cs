
namespace FanucController
{
    partial class MainForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.buttonRunMain = new System.Windows.Forms.Button();
            this.buttonVxReset = new System.Windows.Forms.Button();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.buttonTest = new System.Windows.Forms.Button();
            this.buttonVxStartTracking = new System.Windows.Forms.Button();
            this.buttonVxStopTracking = new System.Windows.Forms.Button();
            this.buttonVxQuickConnect = new System.Windows.Forms.Button();
            this.textBoxVxX = new System.Windows.Forms.TextBox();
            this.textBoxVxY = new System.Windows.Forms.TextBox();
            this.textBoxVxZ = new System.Windows.Forms.TextBox();
            this.textBoxVxAlpha = new System.Windows.Forms.TextBox();
            this.textBoxVxBeta = new System.Windows.Forms.TextBox();
            this.textBoxVxGamma = new System.Windows.Forms.TextBox();
            this.groupBoxVx = new System.Windows.Forms.GroupBox();
            this.textBoxDebug = new System.Windows.Forms.TextBox();
            this.listViewLogger = new System.Windows.Forms.ListView();
            this.columnHeaderLevel = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.columnHeaderMessage = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.labelLog = new System.Windows.Forms.Label();
            this.labelDebug = new System.Windows.Forms.Label();
            this.buttonVxExit = new System.Windows.Forms.Button();
            this.groupBoxPCDK = new System.Windows.Forms.GroupBox();
            this.groupBoxExperiment = new System.Windows.Forms.GroupBox();
            this.groupBoxVx.SuspendLayout();
            this.groupBoxExperiment.SuspendLayout();
            this.SuspendLayout();
            // 
            // buttonRunMain
            // 
            this.buttonRunMain.Location = new System.Drawing.Point(27, 256);
            this.buttonRunMain.Name = "buttonRunMain";
            this.buttonRunMain.Size = new System.Drawing.Size(75, 23);
            this.buttonRunMain.TabIndex = 0;
            this.buttonRunMain.Text = "Run Main";
            this.buttonRunMain.UseVisualStyleBackColor = true;
            this.buttonRunMain.Click += new System.EventHandler(this.buttonRunMain_Click);
            // 
            // buttonVxReset
            // 
            this.buttonVxReset.Location = new System.Drawing.Point(15, 145);
            this.buttonVxReset.Name = "buttonVxReset";
            this.buttonVxReset.Size = new System.Drawing.Size(93, 30);
            this.buttonVxReset.TabIndex = 5;
            this.buttonVxReset.Text = "Reset VXtrack";
            this.buttonVxReset.UseVisualStyleBackColor = true;
            this.buttonVxReset.Click += new System.EventHandler(this.buttonResetVXtrack_Click);
            // 
            // buttonTest
            // 
            this.buttonTest.Location = new System.Drawing.Point(12, 362);
            this.buttonTest.Name = "buttonTest";
            this.buttonTest.Size = new System.Drawing.Size(60, 26);
            this.buttonTest.TabIndex = 6;
            this.buttonTest.Text = "Test";
            this.buttonTest.UseVisualStyleBackColor = true;
            this.buttonTest.Click += new System.EventHandler(this.buttonTest_Click);
            // 
            // buttonVxStartTracking
            // 
            this.buttonVxStartTracking.Location = new System.Drawing.Point(15, 61);
            this.buttonVxStartTracking.Name = "buttonVxStartTracking";
            this.buttonVxStartTracking.Size = new System.Drawing.Size(93, 30);
            this.buttonVxStartTracking.TabIndex = 7;
            this.buttonVxStartTracking.Text = "Start Tracking";
            this.buttonVxStartTracking.UseVisualStyleBackColor = true;
            // 
            // buttonVxStopTracking
            // 
            this.buttonVxStopTracking.Location = new System.Drawing.Point(15, 103);
            this.buttonVxStopTracking.Name = "buttonVxStopTracking";
            this.buttonVxStopTracking.Size = new System.Drawing.Size(93, 30);
            this.buttonVxStopTracking.TabIndex = 7;
            this.buttonVxStopTracking.Text = "Stop Tracking";
            this.buttonVxStopTracking.UseVisualStyleBackColor = true;
            // 
            // buttonVxQuickConnect
            // 
            this.buttonVxQuickConnect.Location = new System.Drawing.Point(15, 19);
            this.buttonVxQuickConnect.Name = "buttonVxQuickConnect";
            this.buttonVxQuickConnect.Size = new System.Drawing.Size(93, 30);
            this.buttonVxQuickConnect.TabIndex = 5;
            this.buttonVxQuickConnect.Text = "Quick Connect";
            this.buttonVxQuickConnect.UseVisualStyleBackColor = true;
            this.buttonVxQuickConnect.Click += new System.EventHandler(this.buttonVxQuickConnect_Click);
            // 
            // textBoxVxX
            // 
            this.textBoxVxX.Location = new System.Drawing.Point(18, 230);
            this.textBoxVxX.Name = "textBoxVxX";
            this.textBoxVxX.Size = new System.Drawing.Size(37, 20);
            this.textBoxVxX.TabIndex = 8;
            // 
            // textBoxVxY
            // 
            this.textBoxVxY.Location = new System.Drawing.Point(75, 230);
            this.textBoxVxY.Name = "textBoxVxY";
            this.textBoxVxY.Size = new System.Drawing.Size(37, 20);
            this.textBoxVxY.TabIndex = 8;
            // 
            // textBoxVxZ
            // 
            this.textBoxVxZ.Location = new System.Drawing.Point(132, 230);
            this.textBoxVxZ.Name = "textBoxVxZ";
            this.textBoxVxZ.Size = new System.Drawing.Size(37, 20);
            this.textBoxVxZ.TabIndex = 8;
            // 
            // textBoxVxAlpha
            // 
            this.textBoxVxAlpha.Location = new System.Drawing.Point(18, 256);
            this.textBoxVxAlpha.Name = "textBoxVxAlpha";
            this.textBoxVxAlpha.Size = new System.Drawing.Size(37, 20);
            this.textBoxVxAlpha.TabIndex = 8;
            // 
            // textBoxVxBeta
            // 
            this.textBoxVxBeta.Location = new System.Drawing.Point(75, 256);
            this.textBoxVxBeta.Name = "textBoxVxBeta";
            this.textBoxVxBeta.Size = new System.Drawing.Size(37, 20);
            this.textBoxVxBeta.TabIndex = 8;
            // 
            // textBoxVxGamma
            // 
            this.textBoxVxGamma.Location = new System.Drawing.Point(132, 256);
            this.textBoxVxGamma.Name = "textBoxVxGamma";
            this.textBoxVxGamma.Size = new System.Drawing.Size(37, 20);
            this.textBoxVxGamma.TabIndex = 8;
            // 
            // groupBoxVx
            // 
            this.groupBoxVx.Controls.Add(this.buttonVxStartTracking);
            this.groupBoxVx.Controls.Add(this.textBoxVxGamma);
            this.groupBoxVx.Controls.Add(this.buttonVxExit);
            this.groupBoxVx.Controls.Add(this.buttonVxReset);
            this.groupBoxVx.Controls.Add(this.textBoxVxBeta);
            this.groupBoxVx.Controls.Add(this.buttonVxQuickConnect);
            this.groupBoxVx.Controls.Add(this.textBoxVxAlpha);
            this.groupBoxVx.Controls.Add(this.buttonVxStopTracking);
            this.groupBoxVx.Controls.Add(this.textBoxVxZ);
            this.groupBoxVx.Controls.Add(this.textBoxVxY);
            this.groupBoxVx.Controls.Add(this.textBoxVxX);
            this.groupBoxVx.Location = new System.Drawing.Point(596, 30);
            this.groupBoxVx.Name = "groupBoxVx";
            this.groupBoxVx.Size = new System.Drawing.Size(184, 299);
            this.groupBoxVx.TabIndex = 9;
            this.groupBoxVx.TabStop = false;
            this.groupBoxVx.Text = "VXelemets";
            // 
            // textBoxDebug
            // 
            this.textBoxDebug.Location = new System.Drawing.Point(573, 416);
            this.textBoxDebug.Multiline = true;
            this.textBoxDebug.Name = "textBoxDebug";
            this.textBoxDebug.Size = new System.Drawing.Size(252, 145);
            this.textBoxDebug.TabIndex = 10;
            // 
            // listViewLogger
            // 
            this.listViewLogger.BackColor = System.Drawing.Color.White;
            this.listViewLogger.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeaderLevel,
            this.columnHeaderMessage});
            this.listViewLogger.GridLines = true;
            this.listViewLogger.HideSelection = false;
            this.listViewLogger.Location = new System.Drawing.Point(12, 416);
            this.listViewLogger.Name = "listViewLogger";
            this.listViewLogger.Size = new System.Drawing.Size(541, 145);
            this.listViewLogger.TabIndex = 11;
            this.listViewLogger.UseCompatibleStateImageBehavior = false;
            this.listViewLogger.View = System.Windows.Forms.View.Details;
            this.listViewLogger.SelectedIndexChanged += new System.EventHandler(this.listViewLogger_SelectedIndexChanged);
            // 
            // columnHeaderLevel
            // 
            this.columnHeaderLevel.Text = "Level";
            // 
            // columnHeaderMessage
            // 
            this.columnHeaderMessage.Text = "Comment";
            this.columnHeaderMessage.Width = 459;
            // 
            // labelLog
            // 
            this.labelLog.AutoSize = true;
            this.labelLog.Location = new System.Drawing.Point(9, 400);
            this.labelLog.Name = "labelLog";
            this.labelLog.Size = new System.Drawing.Size(25, 13);
            this.labelLog.TabIndex = 12;
            this.labelLog.Text = "Log";
            // 
            // labelDebug
            // 
            this.labelDebug.AutoSize = true;
            this.labelDebug.Location = new System.Drawing.Point(570, 400);
            this.labelDebug.Name = "labelDebug";
            this.labelDebug.Size = new System.Drawing.Size(39, 13);
            this.labelDebug.TabIndex = 12;
            this.labelDebug.Text = "Debug";
            // 
            // buttonVxExit
            // 
            this.buttonVxExit.Location = new System.Drawing.Point(15, 181);
            this.buttonVxExit.Name = "buttonVxExit";
            this.buttonVxExit.Size = new System.Drawing.Size(93, 30);
            this.buttonVxExit.TabIndex = 5;
            this.buttonVxExit.Text = "Exit";
            this.buttonVxExit.UseVisualStyleBackColor = true;
            this.buttonVxExit.Click += new System.EventHandler(this.buttonResetVXtrack_Click);
            // 
            // groupBoxPCDK
            // 
            this.groupBoxPCDK.Location = new System.Drawing.Point(304, 30);
            this.groupBoxPCDK.Name = "groupBoxPCDK";
            this.groupBoxPCDK.Size = new System.Drawing.Size(249, 299);
            this.groupBoxPCDK.TabIndex = 13;
            this.groupBoxPCDK.TabStop = false;
            this.groupBoxPCDK.Text = "PCDK";
            // 
            // groupBoxExperiment
            // 
            this.groupBoxExperiment.Controls.Add(this.buttonRunMain);
            this.groupBoxExperiment.Location = new System.Drawing.Point(21, 30);
            this.groupBoxExperiment.Name = "groupBoxExperiment";
            this.groupBoxExperiment.Size = new System.Drawing.Size(249, 299);
            this.groupBoxExperiment.TabIndex = 14;
            this.groupBoxExperiment.TabStop = false;
            this.groupBoxExperiment.Text = "FANUC Experiment";
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(837, 573);
            this.Controls.Add(this.groupBoxExperiment);
            this.Controls.Add(this.groupBoxPCDK);
            this.Controls.Add(this.labelDebug);
            this.Controls.Add(this.labelLog);
            this.Controls.Add(this.listViewLogger);
            this.Controls.Add(this.textBoxDebug);
            this.Controls.Add(this.groupBoxVx);
            this.Controls.Add(this.buttonTest);
            this.Name = "MainForm";
            this.Text = "Form1";
            this.groupBoxVx.ResumeLayout(false);
            this.groupBoxVx.PerformLayout();
            this.groupBoxExperiment.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button buttonRunMain;
        private System.Windows.Forms.Button buttonVxReset;
        private System.Windows.Forms.Timer timer1;
        private System.Windows.Forms.Button buttonTest;
        private System.Windows.Forms.Button buttonVxStartTracking;
        private System.Windows.Forms.Button buttonVxStopTracking;
        private System.Windows.Forms.Button buttonVxQuickConnect;
        private System.Windows.Forms.TextBox textBoxVxX;
        private System.Windows.Forms.TextBox textBoxVxY;
        private System.Windows.Forms.TextBox textBoxVxZ;
        private System.Windows.Forms.TextBox textBoxVxAlpha;
        private System.Windows.Forms.TextBox textBoxVxBeta;
        private System.Windows.Forms.TextBox textBoxVxGamma;
        private System.Windows.Forms.GroupBox groupBoxVx;
        private System.Windows.Forms.TextBox textBoxDebug;
        private System.Windows.Forms.ListView listViewLogger;
        private System.Windows.Forms.ColumnHeader columnHeaderLevel;
        private System.Windows.Forms.ColumnHeader columnHeaderMessage;
        private System.Windows.Forms.Label labelLog;
        private System.Windows.Forms.Label labelDebug;
        private System.Windows.Forms.Button buttonVxExit;
        private System.Windows.Forms.GroupBox groupBoxPCDK;
        private System.Windows.Forms.GroupBox groupBoxExperiment;
    }
}

