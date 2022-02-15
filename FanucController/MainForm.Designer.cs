
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
            this.buttonRunMain = new System.Windows.Forms.Button();
            this.textBox1 = new System.Windows.Forms.TextBox();
            this.textBox2 = new System.Windows.Forms.TextBox();
            this.buttonClearSequences = new System.Windows.Forms.Button();
            this.buttonClearModels = new System.Windows.Forms.Button();
            this.buttonResetVXtrack = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // buttonRunMain
            // 
            this.buttonRunMain.Location = new System.Drawing.Point(153, 151);
            this.buttonRunMain.Name = "buttonRunMain";
            this.buttonRunMain.Size = new System.Drawing.Size(75, 23);
            this.buttonRunMain.TabIndex = 0;
            this.buttonRunMain.Text = "Run Main";
            this.buttonRunMain.UseVisualStyleBackColor = true;
            this.buttonRunMain.Click += new System.EventHandler(this.button1_Click);
            // 
            // textBox1
            // 
            this.textBox1.Location = new System.Drawing.Point(317, 241);
            this.textBox1.Multiline = true;
            this.textBox1.Name = "textBox1";
            this.textBox1.Size = new System.Drawing.Size(149, 85);
            this.textBox1.TabIndex = 1;
            // 
            // textBox2
            // 
            this.textBox2.Location = new System.Drawing.Point(854, 232);
            this.textBox2.Name = "textBox2";
            this.textBox2.Size = new System.Drawing.Size(100, 20);
            this.textBox2.TabIndex = 2;
            this.textBox2.TextChanged += new System.EventHandler(this.textBox2_TextChanged);
            // 
            // buttonClearSequences
            // 
            this.buttonClearSequences.Location = new System.Drawing.Point(591, 180);
            this.buttonClearSequences.Name = "buttonClearSequences";
            this.buttonClearSequences.Size = new System.Drawing.Size(118, 36);
            this.buttonClearSequences.TabIndex = 3;
            this.buttonClearSequences.Text = "Clear Sequence";
            this.buttonClearSequences.UseVisualStyleBackColor = true;
            // 
            // buttonClearModels
            // 
            this.buttonClearModels.Location = new System.Drawing.Point(591, 241);
            this.buttonClearModels.Name = "buttonClearModels";
            this.buttonClearModels.Size = new System.Drawing.Size(118, 36);
            this.buttonClearModels.TabIndex = 4;
            this.buttonClearModels.Text = "Clear Models";
            this.buttonClearModels.UseVisualStyleBackColor = true;
            // 
            // buttonResetVXtrack
            // 
            this.buttonResetVXtrack.Location = new System.Drawing.Point(591, 301);
            this.buttonResetVXtrack.Name = "buttonResetVXtrack";
            this.buttonResetVXtrack.Size = new System.Drawing.Size(118, 36);
            this.buttonResetVXtrack.TabIndex = 5;
            this.buttonResetVXtrack.Text = "Reset VXtrack";
            this.buttonResetVXtrack.UseVisualStyleBackColor = true;
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(837, 573);
            this.Controls.Add(this.buttonResetVXtrack);
            this.Controls.Add(this.buttonClearModels);
            this.Controls.Add(this.buttonClearSequences);
            this.Controls.Add(this.textBox2);
            this.Controls.Add(this.textBox1);
            this.Controls.Add(this.buttonRunMain);
            this.Name = "MainForm";
            this.Text = "Form1";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button buttonRunMain;
        private System.Windows.Forms.TextBox textBox1;
        private System.Windows.Forms.TextBox textBox2;
        private System.Windows.Forms.Button buttonClearSequences;
        private System.Windows.Forms.Button buttonClearModels;
        private System.Windows.Forms.Button buttonResetVXtrack;
    }
}

