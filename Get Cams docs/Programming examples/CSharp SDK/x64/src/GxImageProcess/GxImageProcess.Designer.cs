namespace GxImageProcess
{
    partial class GxImageProcess
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
            this.m_pic_ShowImage = new System.Windows.Forms.PictureBox();
            this.m_ScrollBar_Gamma = new System.Windows.Forms.HScrollBar();
            this.m_txt_Gamma = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.m_CheckBox__BadPixelCorrect = new System.Windows.Forms.CheckBox();
            this.m_ScrollBar_Sharpen = new System.Windows.Forms.HScrollBar();
            this.m_txt_Sharpen = new System.Windows.Forms.TextBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.m_btn_StopDevice = new System.Windows.Forms.Button();
            this.m_btn_StartDevice = new System.Windows.Forms.Button();
            this.m_btn_OpenDevice = new System.Windows.Forms.Button();
            this.m_btn_CloseDevice = new System.Windows.Forms.Button();
            this.m_cb_AwbLight = new System.Windows.Forms.ComboBox();
            this.m_cb_AutoWhite = new System.Windows.Forms.ComboBox();
            this.groupBox7 = new System.Windows.Forms.GroupBox();
            this.m_checkBox_OpenColorCorrect = new System.Windows.Forms.CheckBox();
            this.m_ScrollBar_Saturation = new System.Windows.Forms.HScrollBar();
            this.m_lbl_IsAccelerate = new System.Windows.Forms.Label();
            this.m_checkBox_Accelerate = new System.Windows.Forms.CheckBox();
            this.m_checkBox_OpenDenoise = new System.Windows.Forms.CheckBox();
            this.m_txt_Saturation = new System.Windows.Forms.TextBox();
            this.m_ScrollBar_Lightness = new System.Windows.Forms.HScrollBar();
            this.label11 = new System.Windows.Forms.Label();
            this.m_ScrollBar_Contrast = new System.Windows.Forms.HScrollBar();
            this.label12 = new System.Windows.Forms.Label();
            this.m_txt_Lightness = new System.Windows.Forms.TextBox();
            this.m_txt_Contrast = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.m_checkBox_OpenSharpen = new System.Windows.Forms.CheckBox();
            this.label10 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.label17 = new System.Windows.Forms.Label();
            this.label16 = new System.Windows.Forms.Label();
            this.label15 = new System.Windows.Forms.Label();
            this.label14 = new System.Windows.Forms.Label();
            this.label18 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.m_timer_UpdateAutoWhite = new System.Windows.Forms.Timer(this.components);
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.m_cb_ChooseDevice = new System.Windows.Forms.ComboBox();
            this.label13 = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.m_pic_ShowImage)).BeginInit();
            this.groupBox2.SuspendLayout();
            this.groupBox7.SuspendLayout();
            this.groupBox1.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.groupBox4.SuspendLayout();
            this.SuspendLayout();
            // 
            // m_pic_ShowImage
            // 
            this.m_pic_ShowImage.BackColor = System.Drawing.SystemColors.GradientActiveCaption;
            this.m_pic_ShowImage.Dock = System.Windows.Forms.DockStyle.Left;
            this.m_pic_ShowImage.Location = new System.Drawing.Point(0, 0);
            this.m_pic_ShowImage.Name = "m_pic_ShowImage";
            this.m_pic_ShowImage.Size = new System.Drawing.Size(548, 562);
            this.m_pic_ShowImage.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.m_pic_ShowImage.TabIndex = 25;
            this.m_pic_ShowImage.TabStop = false;
            // 
            // m_ScrollBar_Gamma
            // 
            this.m_ScrollBar_Gamma.LargeChange = 1;
            this.m_ScrollBar_Gamma.Location = new System.Drawing.Point(84, 186);
            this.m_ScrollBar_Gamma.Minimum = 1;
            this.m_ScrollBar_Gamma.Name = "m_ScrollBar_Gamma";
            this.m_ScrollBar_Gamma.Size = new System.Drawing.Size(182, 18);
            this.m_ScrollBar_Gamma.TabIndex = 0;
            this.m_ScrollBar_Gamma.TabStop = true;
            this.m_ScrollBar_Gamma.Value = 1;
            this.m_ScrollBar_Gamma.ValueChanged += new System.EventHandler(this.m_ScrollBar_Gamma_ValueChanged);
            // 
            // m_txt_Gamma
            // 
            this.m_txt_Gamma.Location = new System.Drawing.Point(272, 186);
            this.m_txt_Gamma.Name = "m_txt_Gamma";
            this.m_txt_Gamma.ReadOnly = true;
            this.m_txt_Gamma.Size = new System.Drawing.Size(29, 21);
            this.m_txt_Gamma.TabIndex = 1;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(254, 172);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(17, 12);
            this.label4.TabIndex = 42;
            this.label4.Text = "10";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(86, 173);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(23, 12);
            this.label6.TabIndex = 41;
            this.label6.Text = "0.1";
            // 
            // m_CheckBox__BadPixelCorrect
            // 
            this.m_CheckBox__BadPixelCorrect.AutoSize = true;
            this.m_CheckBox__BadPixelCorrect.Location = new System.Drawing.Point(8, 66);
            this.m_CheckBox__BadPixelCorrect.Name = "m_CheckBox__BadPixelCorrect";
            this.m_CheckBox__BadPixelCorrect.Size = new System.Drawing.Size(168, 16);
            this.m_CheckBox__BadPixelCorrect.TabIndex = 0;
            this.m_CheckBox__BadPixelCorrect.Text = "Defect Correction Enable";
            this.m_CheckBox__BadPixelCorrect.UseVisualStyleBackColor = true;
            this.m_CheckBox__BadPixelCorrect.CheckedChanged += new System.EventHandler(this.m_CheckBox__BadPixelCorrect_CheckedChanged);
            // 
            // m_ScrollBar_Sharpen
            // 
            this.m_ScrollBar_Sharpen.LargeChange = 1;
            this.m_ScrollBar_Sharpen.Location = new System.Drawing.Point(84, 152);
            this.m_ScrollBar_Sharpen.Maximum = 50;
            this.m_ScrollBar_Sharpen.Minimum = 1;
            this.m_ScrollBar_Sharpen.Name = "m_ScrollBar_Sharpen";
            this.m_ScrollBar_Sharpen.Size = new System.Drawing.Size(182, 18);
            this.m_ScrollBar_Sharpen.TabIndex = 1;
            this.m_ScrollBar_Sharpen.TabStop = true;
            this.m_ScrollBar_Sharpen.Value = 1;
            this.m_ScrollBar_Sharpen.ValueChanged += new System.EventHandler(this.m_ScrollBar_Sharpen_ValueChanged);
            // 
            // m_txt_Sharpen
            // 
            this.m_txt_Sharpen.Location = new System.Drawing.Point(272, 149);
            this.m_txt_Sharpen.Name = "m_txt_Sharpen";
            this.m_txt_Sharpen.ReadOnly = true;
            this.m_txt_Sharpen.Size = new System.Drawing.Size(29, 21);
            this.m_txt_Sharpen.TabIndex = 2;
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.m_btn_StopDevice);
            this.groupBox2.Controls.Add(this.m_btn_StartDevice);
            this.groupBox2.Controls.Add(this.m_btn_OpenDevice);
            this.groupBox2.Controls.Add(this.m_btn_CloseDevice);
            this.groupBox2.Location = new System.Drawing.Point(554, 12);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(369, 78);
            this.groupBox2.TabIndex = 1;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Device Control";
            // 
            // m_btn_StopDevice
            // 
            this.m_btn_StopDevice.Location = new System.Drawing.Point(221, 46);
            this.m_btn_StopDevice.Name = "m_btn_StopDevice";
            this.m_btn_StopDevice.Size = new System.Drawing.Size(130, 23);
            this.m_btn_StopDevice.TabIndex = 3;
            this.m_btn_StopDevice.Text = "Acquisition Stop";
            this.m_btn_StopDevice.UseVisualStyleBackColor = true;
            this.m_btn_StopDevice.Click += new System.EventHandler(this.m_btn_StopDevice_Click);
            // 
            // m_btn_StartDevice
            // 
            this.m_btn_StartDevice.Location = new System.Drawing.Point(25, 47);
            this.m_btn_StartDevice.Name = "m_btn_StartDevice";
            this.m_btn_StartDevice.Size = new System.Drawing.Size(130, 23);
            this.m_btn_StartDevice.TabIndex = 2;
            this.m_btn_StartDevice.Text = "Acquisition Start";
            this.m_btn_StartDevice.UseVisualStyleBackColor = true;
            this.m_btn_StartDevice.Click += new System.EventHandler(this.m_btn_StartDevice_Click);
            // 
            // m_btn_OpenDevice
            // 
            this.m_btn_OpenDevice.Location = new System.Drawing.Point(25, 16);
            this.m_btn_OpenDevice.Name = "m_btn_OpenDevice";
            this.m_btn_OpenDevice.Size = new System.Drawing.Size(130, 23);
            this.m_btn_OpenDevice.TabIndex = 0;
            this.m_btn_OpenDevice.Text = "Open Device";
            this.m_btn_OpenDevice.UseVisualStyleBackColor = true;
            this.m_btn_OpenDevice.Click += new System.EventHandler(this.m_btn_OpenDevice_Click);
            // 
            // m_btn_CloseDevice
            // 
            this.m_btn_CloseDevice.Location = new System.Drawing.Point(221, 16);
            this.m_btn_CloseDevice.Name = "m_btn_CloseDevice";
            this.m_btn_CloseDevice.Size = new System.Drawing.Size(130, 23);
            this.m_btn_CloseDevice.TabIndex = 1;
            this.m_btn_CloseDevice.Text = "Close Device";
            this.m_btn_CloseDevice.UseVisualStyleBackColor = true;
            this.m_btn_CloseDevice.Click += new System.EventHandler(this.m_btn_CloseDevice_Click);
            // 
            // m_cb_AwbLight
            // 
            this.m_cb_AwbLight.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.m_cb_AwbLight.FormattingEnabled = true;
            this.m_cb_AwbLight.Location = new System.Drawing.Point(4, 20);
            this.m_cb_AwbLight.Name = "m_cb_AwbLight";
            this.m_cb_AwbLight.Size = new System.Drawing.Size(130, 20);
            this.m_cb_AwbLight.TabIndex = 3;
            this.m_cb_AwbLight.SelectedIndexChanged += new System.EventHandler(this.m_cb_AwbLight_SelectedIndexChanged);
            // 
            // m_cb_AutoWhite
            // 
            this.m_cb_AutoWhite.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.m_cb_AutoWhite.FormattingEnabled = true;
            this.m_cb_AutoWhite.Location = new System.Drawing.Point(8, 20);
            this.m_cb_AutoWhite.Name = "m_cb_AutoWhite";
            this.m_cb_AutoWhite.Size = new System.Drawing.Size(130, 20);
            this.m_cb_AutoWhite.TabIndex = 1;
            this.m_cb_AutoWhite.SelectedIndexChanged += new System.EventHandler(this.m_cb_AutoWhite_SelectedIndexChanged);
            // 
            // groupBox7
            // 
            this.groupBox7.Controls.Add(this.m_checkBox_OpenColorCorrect);
            this.groupBox7.Controls.Add(this.m_ScrollBar_Saturation);
            this.groupBox7.Controls.Add(this.m_lbl_IsAccelerate);
            this.groupBox7.Controls.Add(this.m_CheckBox__BadPixelCorrect);
            this.groupBox7.Controls.Add(this.m_checkBox_Accelerate);
            this.groupBox7.Controls.Add(this.m_checkBox_OpenDenoise);
            this.groupBox7.Controls.Add(this.m_txt_Saturation);
            this.groupBox7.Controls.Add(this.m_ScrollBar_Lightness);
            this.groupBox7.Controls.Add(this.label11);
            this.groupBox7.Controls.Add(this.m_ScrollBar_Contrast);
            this.groupBox7.Controls.Add(this.label12);
            this.groupBox7.Controls.Add(this.m_txt_Lightness);
            this.groupBox7.Controls.Add(this.label4);
            this.groupBox7.Controls.Add(this.m_txt_Contrast);
            this.groupBox7.Controls.Add(this.label9);
            this.groupBox7.Controls.Add(this.m_checkBox_OpenSharpen);
            this.groupBox7.Controls.Add(this.label6);
            this.groupBox7.Controls.Add(this.label10);
            this.groupBox7.Controls.Add(this.label7);
            this.groupBox7.Controls.Add(this.m_txt_Gamma);
            this.groupBox7.Controls.Add(this.m_txt_Sharpen);
            this.groupBox7.Controls.Add(this.label8);
            this.groupBox7.Controls.Add(this.m_ScrollBar_Sharpen);
            this.groupBox7.Controls.Add(this.m_ScrollBar_Gamma);
            this.groupBox7.Controls.Add(this.label17);
            this.groupBox7.Controls.Add(this.label16);
            this.groupBox7.Controls.Add(this.label15);
            this.groupBox7.Controls.Add(this.label14);
            this.groupBox7.Controls.Add(this.label13);
            this.groupBox7.Controls.Add(this.label18);
            this.groupBox7.Controls.Add(this.label3);
            this.groupBox7.Controls.Add(this.label2);
            this.groupBox7.Location = new System.Drawing.Point(554, 224);
            this.groupBox7.Name = "groupBox7";
            this.groupBox7.Size = new System.Drawing.Size(369, 338);
            this.groupBox7.TabIndex = 7;
            this.groupBox7.TabStop = false;
            this.groupBox7.Text = "Image Process";
            // 
            // m_checkBox_OpenColorCorrect
            // 
            this.m_checkBox_OpenColorCorrect.AutoSize = true;
            this.m_checkBox_OpenColorCorrect.Location = new System.Drawing.Point(8, 120);
            this.m_checkBox_OpenColorCorrect.Name = "m_checkBox_OpenColorCorrect";
            this.m_checkBox_OpenColorCorrect.Size = new System.Drawing.Size(162, 16);
            this.m_checkBox_OpenColorCorrect.TabIndex = 0;
            this.m_checkBox_OpenColorCorrect.Text = "Color Correction Enable";
            this.m_checkBox_OpenColorCorrect.UseVisualStyleBackColor = true;
            this.m_checkBox_OpenColorCorrect.CheckedChanged += new System.EventHandler(this.m_checkBox_OpenColorCorrect_CheckedChanged);
            // 
            // m_ScrollBar_Saturation
            // 
            this.m_ScrollBar_Saturation.LargeChange = 1;
            this.m_ScrollBar_Saturation.Location = new System.Drawing.Point(84, 309);
            this.m_ScrollBar_Saturation.Maximum = 128;
            this.m_ScrollBar_Saturation.Name = "m_ScrollBar_Saturation";
            this.m_ScrollBar_Saturation.Size = new System.Drawing.Size(182, 18);
            this.m_ScrollBar_Saturation.TabIndex = 0;
            this.m_ScrollBar_Saturation.TabStop = true;
            this.m_ScrollBar_Saturation.ValueChanged += new System.EventHandler(this.m_ScrollBar_Saturation_ValueChanged);
            // 
            // m_lbl_IsAccelerate
            // 
            this.m_lbl_IsAccelerate.AutoSize = true;
            this.m_lbl_IsAccelerate.Location = new System.Drawing.Point(25, 41);
            this.m_lbl_IsAccelerate.Name = "m_lbl_IsAccelerate";
            this.m_lbl_IsAccelerate.Size = new System.Drawing.Size(245, 12);
            this.m_lbl_IsAccelerate.TabIndex = 0;
            this.m_lbl_IsAccelerate.Text = "(Need CPU support SSSE3 instruction set)";
            // 
            // m_checkBox_Accelerate
            // 
            this.m_checkBox_Accelerate.AutoSize = true;
            this.m_checkBox_Accelerate.Location = new System.Drawing.Point(8, 25);
            this.m_checkBox_Accelerate.Name = "m_checkBox_Accelerate";
            this.m_checkBox_Accelerate.Size = new System.Drawing.Size(198, 16);
            this.m_checkBox_Accelerate.TabIndex = 1;
            this.m_checkBox_Accelerate.Text = "Algorithm Acceleration Enable";
            this.m_checkBox_Accelerate.UseVisualStyleBackColor = true;
            this.m_checkBox_Accelerate.CheckedChanged += new System.EventHandler(this.m_checkBox_Accelerate_CheckedChanged);
            // 
            // m_checkBox_OpenDenoise
            // 
            this.m_checkBox_OpenDenoise.AutoSize = true;
            this.m_checkBox_OpenDenoise.Location = new System.Drawing.Point(8, 93);
            this.m_checkBox_OpenDenoise.Name = "m_checkBox_OpenDenoise";
            this.m_checkBox_OpenDenoise.Size = new System.Drawing.Size(156, 16);
            this.m_checkBox_OpenDenoise.TabIndex = 0;
            this.m_checkBox_OpenDenoise.Text = "Image Denoising Enable";
            this.m_checkBox_OpenDenoise.UseVisualStyleBackColor = true;
            this.m_checkBox_OpenDenoise.CheckedChanged += new System.EventHandler(this.m_checkBox_OpenDenoise_CheckedChanged);
            // 
            // m_txt_Saturation
            // 
            this.m_txt_Saturation.Location = new System.Drawing.Point(272, 305);
            this.m_txt_Saturation.Name = "m_txt_Saturation";
            this.m_txt_Saturation.ReadOnly = true;
            this.m_txt_Saturation.Size = new System.Drawing.Size(29, 21);
            this.m_txt_Saturation.TabIndex = 1;
            // 
            // m_ScrollBar_Lightness
            // 
            this.m_ScrollBar_Lightness.LargeChange = 1;
            this.m_ScrollBar_Lightness.Location = new System.Drawing.Point(84, 267);
            this.m_ScrollBar_Lightness.Maximum = 150;
            this.m_ScrollBar_Lightness.Minimum = -150;
            this.m_ScrollBar_Lightness.Name = "m_ScrollBar_Lightness";
            this.m_ScrollBar_Lightness.Size = new System.Drawing.Size(182, 18);
            this.m_ScrollBar_Lightness.TabIndex = 0;
            this.m_ScrollBar_Lightness.TabStop = true;
            this.m_ScrollBar_Lightness.Value = -150;
            this.m_ScrollBar_Lightness.ValueChanged += new System.EventHandler(this.m_ScrollBar_Lightness_ValueChanged);
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(254, 290);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(23, 12);
            this.label11.TabIndex = 54;
            this.label11.Text = "128";
            // 
            // m_ScrollBar_Contrast
            // 
            this.m_ScrollBar_Contrast.LargeChange = 1;
            this.m_ScrollBar_Contrast.Location = new System.Drawing.Point(84, 226);
            this.m_ScrollBar_Contrast.Minimum = -50;
            this.m_ScrollBar_Contrast.Name = "m_ScrollBar_Contrast";
            this.m_ScrollBar_Contrast.Size = new System.Drawing.Size(182, 18);
            this.m_ScrollBar_Contrast.TabIndex = 0;
            this.m_ScrollBar_Contrast.TabStop = true;
            this.m_ScrollBar_Contrast.Value = -50;
            this.m_ScrollBar_Contrast.ValueChanged += new System.EventHandler(this.m_ScrollBar_Contrast_ValueChanged);
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Location = new System.Drawing.Point(86, 290);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(11, 12);
            this.label12.TabIndex = 53;
            this.label12.Text = "0";
            // 
            // m_txt_Lightness
            // 
            this.m_txt_Lightness.Location = new System.Drawing.Point(272, 264);
            this.m_txt_Lightness.Name = "m_txt_Lightness";
            this.m_txt_Lightness.ReadOnly = true;
            this.m_txt_Lightness.Size = new System.Drawing.Size(29, 21);
            this.m_txt_Lightness.TabIndex = 1;
            // 
            // m_txt_Contrast
            // 
            this.m_txt_Contrast.Location = new System.Drawing.Point(272, 223);
            this.m_txt_Contrast.Name = "m_txt_Contrast";
            this.m_txt_Contrast.ReadOnly = true;
            this.m_txt_Contrast.Size = new System.Drawing.Size(29, 21);
            this.m_txt_Contrast.TabIndex = 1;
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(254, 248);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(23, 12);
            this.label9.TabIndex = 50;
            this.label9.Text = "150";
            // 
            // m_checkBox_OpenSharpen
            // 
            this.m_checkBox_OpenSharpen.AutoSize = true;
            this.m_checkBox_OpenSharpen.Location = new System.Drawing.Point(306, 152);
            this.m_checkBox_OpenSharpen.Name = "m_checkBox_OpenSharpen";
            this.m_checkBox_OpenSharpen.Size = new System.Drawing.Size(60, 16);
            this.m_checkBox_OpenSharpen.TabIndex = 0;
            this.m_checkBox_OpenSharpen.Text = "Enable";
            this.m_checkBox_OpenSharpen.UseVisualStyleBackColor = true;
            this.m_checkBox_OpenSharpen.CheckedChanged += new System.EventHandler(this.m_checkBox_OpenSharpen_CheckedChanged);
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(86, 248);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(29, 12);
            this.label10.TabIndex = 49;
            this.label10.Text = "-150";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(254, 209);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(23, 12);
            this.label7.TabIndex = 46;
            this.label7.Text = "100";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(86, 212);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(23, 12);
            this.label8.TabIndex = 45;
            this.label8.Text = "-50";
            // 
            // label17
            // 
            this.label17.AutoSize = true;
            this.label17.Location = new System.Drawing.Point(8, 312);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(65, 12);
            this.label17.TabIndex = 0;
            this.label17.Text = "Saturation";
            // 
            // label16
            // 
            this.label16.AutoSize = true;
            this.label16.Location = new System.Drawing.Point(8, 271);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(59, 12);
            this.label16.TabIndex = 0;
            this.label16.Text = "Lightness";
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.Location = new System.Drawing.Point(8, 229);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(53, 12);
            this.label15.TabIndex = 0;
            this.label15.Text = "Contrast";
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Location = new System.Drawing.Point(8, 189);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(35, 12);
            this.label14.TabIndex = 0;
            this.label14.Text = "Gamma";
            // 
            // label18
            // 
            this.label18.AutoSize = true;
            this.label18.Location = new System.Drawing.Point(254, 137);
            this.label18.Name = "label18";
            this.label18.Size = new System.Drawing.Size(11, 12);
            this.label18.TabIndex = 39;
            this.label18.Text = "5";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(254, 137);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(11, 12);
            this.label3.TabIndex = 39;
            this.label3.Text = "5";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(86, 137);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(23, 12);
            this.label2.TabIndex = 38;
            this.label2.Text = "0.1";
            // 
            // m_timer_UpdateAutoWhite
            // 
            this.m_timer_UpdateAutoWhite.Enabled = true;
            this.m_timer_UpdateAutoWhite.Interval = 1000;
            this.m_timer_UpdateAutoWhite.Tick += new System.EventHandler(this.m_timer_UpdateAutoWhite_Tick);
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.m_cb_AutoWhite);
            this.groupBox1.Location = new System.Drawing.Point(554, 161);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(155, 57);
            this.groupBox1.TabIndex = 4;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "BalanceWhiteAuto";
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.m_cb_AwbLight);
            this.groupBox3.Location = new System.Drawing.Point(768, 161);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(147, 57);
            this.groupBox3.TabIndex = 5;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "AWBLampHouse";
            // 
            // groupBox4
            // 
            this.groupBox4.Controls.Add(this.m_cb_ChooseDevice);
            this.groupBox4.Location = new System.Drawing.Point(554, 98);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Size = new System.Drawing.Size(366, 57);
            this.groupBox4.TabIndex = 26;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "BalanceWhiteAuto";
            // 
            // m_cb_ChooseDevice
            // 
            this.m_cb_ChooseDevice.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.m_cb_ChooseDevice.FormattingEnabled = true;
            this.m_cb_ChooseDevice.Location = new System.Drawing.Point(8, 20);
            this.m_cb_ChooseDevice.Name = "m_cb_ChooseDevice";
            this.m_cb_ChooseDevice.Size = new System.Drawing.Size(340, 20);
            this.m_cb_ChooseDevice.TabIndex = 1;
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Location = new System.Drawing.Point(8, 154);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(47, 12);
            this.label13.TabIndex = 0;
            this.label13.Text = "Sharpen";
            // 
            // GxImageProcess
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(927, 562);
            this.Controls.Add(this.groupBox4);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.m_pic_ShowImage);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox7);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.KeyPreview = true;
            this.MaximizeBox = false;
            this.Name = "GxImageProcess";
            this.Text = "GxImageProcess";
            this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.GxImageProcess_FormClosed);
            this.Load += new System.EventHandler(this.GxImageProcess_Load);
            ((System.ComponentModel.ISupportInitialize)(this.m_pic_ShowImage)).EndInit();
            this.groupBox2.ResumeLayout(false);
            this.groupBox7.ResumeLayout(false);
            this.groupBox7.PerformLayout();
            this.groupBox1.ResumeLayout(false);
            this.groupBox3.ResumeLayout(false);
            this.groupBox4.ResumeLayout(false);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PictureBox m_pic_ShowImage;
        private System.Windows.Forms.TextBox m_txt_Gamma;
        private System.Windows.Forms.CheckBox m_CheckBox__BadPixelCorrect;
        private System.Windows.Forms.HScrollBar m_ScrollBar_Sharpen;
        private System.Windows.Forms.TextBox m_txt_Sharpen;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.Button m_btn_StopDevice;
        private System.Windows.Forms.Button m_btn_StartDevice;
        private System.Windows.Forms.Button m_btn_OpenDevice;
        private System.Windows.Forms.Button m_btn_CloseDevice;
        private System.Windows.Forms.GroupBox groupBox7;
        private System.Windows.Forms.CheckBox m_checkBox_OpenDenoise;
        private System.Windows.Forms.CheckBox m_checkBox_OpenColorCorrect;
        private System.Windows.Forms.CheckBox m_checkBox_OpenSharpen;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.HScrollBar m_ScrollBar_Gamma;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.HScrollBar m_ScrollBar_Contrast;
        private System.Windows.Forms.TextBox m_txt_Contrast;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.HScrollBar m_ScrollBar_Lightness;
        private System.Windows.Forms.TextBox m_txt_Lightness;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.HScrollBar m_ScrollBar_Saturation;
        private System.Windows.Forms.TextBox m_txt_Saturation;
        private System.Windows.Forms.Timer m_timer_UpdateAutoWhite;
        private System.Windows.Forms.ComboBox m_cb_AwbLight;
        private System.Windows.Forms.ComboBox m_cb_AutoWhite;
        private System.Windows.Forms.Label m_lbl_IsAccelerate;
        private System.Windows.Forms.CheckBox m_checkBox_Accelerate;
        private System.Windows.Forms.Label label15;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.Label label17;
        private System.Windows.Forms.Label label16;
        private System.Windows.Forms.Label label18;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.GroupBox groupBox4;
        private System.Windows.Forms.ComboBox m_cb_ChooseDevice;
        private System.Windows.Forms.Label label13;
    }
}

