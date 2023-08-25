using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Threading;
using System.Diagnostics;
using GxIAPINET;
using GxIAPINET.Sample.Common;

namespace GxMultiCam
{
    public partial class GxMultiCam : Form
    {

        List<CCamerInfo> m_listCCamerInfo = new List<CCamerInfo>();                              ///<Camera parameter status list
        IGXFactory                   m_objIGXFactory         = null;                             ///<The object for Factory
        List<IGXDeviceInfo>          m_listIGXDeviceInfo     = new List<IGXDeviceInfo>();        ///<Device information list
        int                          m_nCamNum	             = 0;								 ///<Initialize the number of device 
        int                          m_nOperateID            = -1;                               ///<Device index        

        

        /// <summary>
        /// Construction
        /// </summary>
        public GxMultiCam()
        {
            InitializeComponent();    
        }

        /// <summary>
        ///Initialize UI
        /// </summary>
        private void __LoadInitUI()
        {
            m_btn_OpenDevice.Enabled = false;
            m_btn_CloseDevice.Enabled = false;
            m_btn_StartDevice.Enabled = false;
            m_btn_StopDevice.Enabled =  false;

            m_txt_Shutter.Enabled = false;
            m_txt_Gain.Enabled = false;
            m_txt_AcqSpeedLevel.Enabled = false;
            m_cb_AutoWhite.Enabled = false;
        }

        /// <summary>
        /// Initialize UI after opening device
        /// </summary>
        private void __InitUI()
        {
            __InitShutterUI();                                                                                                   //Exposure initialization
            __InitGainUI();                                                                                                      //Gain initialization
            __InitAcquisitionSpeedLevelUI();                                                                                     //AcquisitionSpeedLeve initialization
            __InitEnumComBoxUI(m_cb_AutoWhite, "BalanceWhiteAuto"
                            , m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl
                            , ref m_listCCamerInfo[m_nOperateID].m_bWhiteAuto);                                                 //BalanceWhiteAuto initialization
            //Get current white balance 
            bool bIsImplemented = false;             //whether can be support
            bool bIsReadable = false;                //whether can be read
            
            if (null != m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl)
            {
                bIsImplemented = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.IsImplemented("BalanceWhiteAuto");
                bIsReadable = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.IsReadable("BalanceWhiteAuto");
                if (bIsImplemented)
                {
                    if (bIsReadable)
                    {
                        
                        m_listCCamerInfo[m_nOperateID].m_strBalanceWhiteAutoValue = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetEnumFeature("BalanceWhiteAuto").GetValue();
                    }
                }
            }
        }

        /// <summary>
        /// Initialize combobox
        /// </summary>
        /// <param name="cbEnum">ComboBox name</param>
        /// <param name="strFeatureName">function name</param>
        /// <param name="objIGXFeatureControl">the object of feature control</param>
        /// <param name="bIsImplemented">Whether support</param>
        private void __InitEnumComBoxUI(ComboBox cbEnum, string strFeatureName, IGXFeatureControl objIGXFeatureControl, ref bool bIsImplemented)
        {
            string strTriggerValue = "";                   
            List<string> list = new List<string>();        
            bool bIsReadable = false;

            // Get support
            if (null != objIGXFeatureControl)
            {

                bIsImplemented = objIGXFeatureControl.IsImplemented(strFeatureName);

                // If not supported, return directly
                if (!bIsImplemented)
                {
                    return;
                }

                bIsReadable = objIGXFeatureControl.IsReadable(strFeatureName);

                if (bIsReadable)
                {
                    list.AddRange(objIGXFeatureControl.GetEnumFeature(strFeatureName).GetEnumEntryList());
                    //Get current function value
                    strTriggerValue = objIGXFeatureControl.GetEnumFeature(strFeatureName).GetValue();
                }

            }

            // Clear combobox
            cbEnum.Items.Clear();
            foreach (string str in list)
            {
                cbEnum.Items.Add(str);
            }

            for (int i = 0; i < cbEnum.Items.Count; i++)
            {
                string strTemp = cbEnum.Items[i].ToString();
                if (strTemp == strTriggerValue)
                {
                    cbEnum.SelectedIndex = i;
                    break;
                }
            }
        }


        /// <summary>
        /// Shows less than 5 cameras
        /// </summary>
        /// <param name="nDeviceCount">The actual number of enumerations</param>
        /// <returns>The minimum number of display</returns>
        private int __GetMin(int nDeviceCount)
        {
            if (nDeviceCount < 0)
            {
                return 0;
            }

            if (nDeviceCount > 4)
            {
                return 4;
            }

            return nDeviceCount;
        }

        /// <summary>
        /// Loading form to perform initialization UI and Library
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void GxMultiCam_Load(object sender, EventArgs e)
        {
            try
            {
                __LoadInitUI();
                m_objIGXFactory = IGXFactory.GetInstance();
                m_objIGXFactory.Init();
                __EnumDevice();     
            }
            catch (Exception)
            {
            }
        }

        /// <summary>
        /// Update UI
        /// </summary>
        private void __UpdateUI()
        {
            bool bIsDevOpened = m_listCCamerInfo[m_nOperateID].m_bIsOpen;
            bool bIsSnap = m_listCCamerInfo[m_nOperateID].m_bIsSnap;
            bool bWhiteAuto = m_listCCamerInfo[m_nOperateID].m_bWhiteAuto;
            bool bAcqSpeedLevel = m_listCCamerInfo[m_nOperateID].m_bAcqSpeedLevel;

            
            m_btn_OpenDevice.Enabled = !bIsDevOpened;
            m_btn_CloseDevice.Enabled = bIsDevOpened;
            m_btn_StartDevice.Enabled = bIsDevOpened && !bIsSnap;
            m_btn_StopDevice.Enabled  = bIsSnap;

            m_txt_Shutter.Enabled = bIsDevOpened;
            m_txt_Gain.Enabled = bIsDevOpened;
            m_txt_AcqSpeedLevel.Enabled = bIsDevOpened && (!bIsSnap) && bAcqSpeedLevel;
            m_cb_AutoWhite.Enabled = bIsDevOpened && bWhiteAuto;       
        }

        /// <summary>
        /// Enum Device
        /// </summary>
        private void __EnumDevice()
        {
            m_listIGXDeviceInfo.Clear();
            if (null != m_objIGXFactory)
            {
                m_objIGXFactory.UpdateDeviceList(200, m_listIGXDeviceInfo);
            }

            if (m_listIGXDeviceInfo.Count <= 0)
            {
                MessageBox.Show("No device found,please make sure the device is properly connected and then restart the program!");
                return;
            }

            m_nCamNum = __GetMin(m_listIGXDeviceInfo.Count);
            m_cb_EnumDevice.Items.Clear();
            for (int i = 0; i < m_nCamNum;i++ )
            {
                m_cb_EnumDevice.Items.Add(m_listIGXDeviceInfo[i].GetDisplayName());
                CCamerInfo objCCamerInfo = new CCamerInfo();
                objCCamerInfo.m_strDisplayName = m_listIGXDeviceInfo[i].GetDisplayName();
                objCCamerInfo.m_strSN = m_listIGXDeviceInfo[i].GetSN();
                objCCamerInfo.m_emDeviceType = m_listIGXDeviceInfo[i].GetDeviceClass();
                m_listCCamerInfo.Add(objCCamerInfo);
            }

            if (m_cb_EnumDevice.Items.Count > 0)
            {
                m_cb_EnumDevice.SelectedIndex = 0;
            }

            for (int i = 0; i < m_nCamNum; i++)
            {
                if (m_cb_EnumDevice.Text == m_listCCamerInfo[i].m_strDisplayName)
                {
                    m_nOperateID = i;
                }
            }
           
            __UpdateUI();
        }

        /// <summary>
        /// Close all device and stream
        /// </summary>
        private void __CloseAll()
        {
            for (int i = 0; i < m_listCCamerInfo.Count; i++)
            {
                if (null != m_listCCamerInfo[m_nOperateID].m_objImageShowFrom)
                {
                    m_listCCamerInfo[m_nOperateID].m_objImageShowFrom.CloseDevice();
                    m_listCCamerInfo[i].m_bIsSnap = false;
                    m_listCCamerInfo[i].m_bIsOpen = false;
                }
            }
            m_listCCamerInfo.Clear();
        }

        /// <summary>
        /// Initialize device param
        /// </summary>
        private void __InitParam()
        {
            if (null != m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl)
            {
                // Set the continuous frame acquisition mode
                m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetEnumFeature("AcquisitionMode").SetValue("Continuous");

                // Set the TriggerMode off
                m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetEnumFeature("TriggerMode").SetValue("Off");
            }
        }

        /// <summary>
        /// Initialize exposure edit
        /// </summary>
        private void __InitShutterUI()
        {
            double dCurShuter = 0.0;                 //current exposure time
            double dMin = 0.0;                       //Min exposure time
            double dMax = 0.0;                       //Max exposure time
            string strUnit = "";                     //exposure time unit
            string strText = "";                     //display information

            if (null != m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl)
            {
                dCurShuter = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetFloatFeature("ExposureTime").GetValue();
                dMin = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetFloatFeature("ExposureTime").GetMin();
                dMax = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetFloatFeature("ExposureTime").GetMax();
                strUnit = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetFloatFeature("ExposureTime").GetUnit();
            }
      
            strText = string.Format("Exposuretime({0}~{1}){2}", dMin.ToString("0.00"), dMax.ToString("0.00"), strUnit);
            m_lbl_Shutter.Text = strText;

            m_txt_Shutter.Text = dCurShuter.ToString("0.00");
        }

        /// <summary>
        /// Initialize gain edit
        /// </summary>
        private void __InitGainUI()
        {
            double dCurGain = 0;             //current gain
            double dMin = 0.0;               //Min gain
            double dMax = 0.0;               //Max gain
            string strUnit = "";             //gain unit
            string strText = "";             //display information

           
            if (null != m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl)
            {
                dCurGain = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetFloatFeature("Gain").GetValue();
                dMin = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetFloatFeature("Gain").GetMin();
                dMax = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetFloatFeature("Gain").GetMax();
                strUnit = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetFloatFeature("Gain").GetUnit();
            }

           
            strText = string.Format("Gain({0}~{1}){2}", dMin.ToString("0.00"), dMax.ToString("0.00"), strUnit);
            m_lbl_Gain.Text = strText;

            
            string strCurGain = dCurGain.ToString("0.00");
            m_txt_Gain.Text = strCurGain;
        }

        /// <summary>
        /// Initialize AcquisitionSpeedLevel edit
        /// </summary>
        private void __InitAcquisitionSpeedLevelUI()
        {
            Int64 nCurAcquisitionSpeedLevel = 0;             //current AcquisitionSpeedLevel
            Int64 nMin = 0;           //Min AcquisitionSpeedLevel
            Int64 nMax = 0;           //Max AcquisitionSpeedLevel
            string strText = "";      //display information

            if (null != m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl)
            {
                if (!(m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.IsImplemented("AcquisitionSpeedLevel")))
                {
                    m_listCCamerInfo[m_nOperateID].m_bAcqSpeedLevel = false;

                    m_lbl_AcqSpeedLevel.Text = "AcquisitionSpeedLevel";
                    m_txt_AcqSpeedLevel.Text = "";
                    return;
                }

                m_listCCamerInfo[m_nOperateID].m_bAcqSpeedLevel = true;
                nCurAcquisitionSpeedLevel = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetIntFeature("AcquisitionSpeedLevel").GetValue();
                nMin = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetIntFeature("AcquisitionSpeedLevel").GetMin();
                nMax = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetIntFeature("AcquisitionSpeedLevel").GetMax();
            }


            strText = string.Format("AcquisitionSpeedLevel({0}~{1})", nMin.ToString(), nMax.ToString());
            m_lbl_AcqSpeedLevel.Text = strText;

            string strCurAcquisitionSpeedLevel = nCurAcquisitionSpeedLevel.ToString();
            m_txt_AcqSpeedLevel.Text = strCurAcquisitionSpeedLevel;
        }

        /// <summary>
        /// Open device
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_btn_OpenDevice_Click(object sender, EventArgs e)
        {
            string strSN = ""; //Device serial name
            try
            {
                // Check whether has device
                if (m_listIGXDeviceInfo.Count <= 0)
                {
                    MessageBox.Show("No device found!");
                    return;
                }

                try
                {
                    //Stop acquisition and close stream
                    if (null != m_listCCamerInfo[m_nOperateID].m_objIGXStream)
                    {
                        m_listCCamerInfo[m_nOperateID].m_objIGXStream.Close();
                        m_listCCamerInfo[m_nOperateID].m_objIGXStream = null;
                    }
                }
                catch (Exception)
                {
                }


                if (null != m_listCCamerInfo[m_nOperateID].m_objIGXDevice)
                {
                    m_listCCamerInfo[m_nOperateID].m_objIGXDevice.Close();
                    m_listCCamerInfo[m_nOperateID].m_objIGXDevice = null;
                }

                strSN = m_listIGXDeviceInfo[m_nOperateID].GetSN();
                IGXDevice objIGXDevice= m_objIGXFactory.OpenDeviceBySN(strSN, GX_ACCESS_MODE.GX_ACCESS_EXCLUSIVE);
                //Open device
                m_listCCamerInfo[m_nOperateID].m_objIGXDevice = objIGXDevice;
                m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl = m_listCCamerInfo[m_nOperateID].m_objIGXDevice.GetRemoteFeatureControl();

                __InitParam();

                // Initialize UI
                __InitUI();

                ImageShowFrom objImageShowFrom = new ImageShowFrom(objIGXDevice, m_nOperateID);
                objImageShowFrom.OpenDevice();
                m_listCCamerInfo[m_nOperateID].m_objImageShowFrom = objImageShowFrom;
                __SelectDeviceAndShow(m_nOperateID);

                m_listCCamerInfo[m_nOperateID].m_bIsOpen = true;

                //Update UI
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// According to the corresponding device object binding display control
        /// </summary>
        /// <param name="nCamID">Device ID</param>
        private void __SelectDeviceAndShow(int nCamID)
        {
            switch (nCamID)
            {
                case 0:
                    {
                        m_listCCamerInfo[nCamID].m_objImageShowFrom.MdiParent = this;
                        m_listCCamerInfo[nCamID].m_objImageShowFrom.StartPosition = FormStartPosition.Manual;
                        m_listCCamerInfo[nCamID].m_objImageShowFrom.Location = new Point(0, 0);
                        m_listCCamerInfo[nCamID].m_objImageShowFrom.Show();
                    }
                    break;
                case 1:
                    {
                        m_listCCamerInfo[nCamID].m_objImageShowFrom.MdiParent = this;
                        m_listCCamerInfo[nCamID].m_objImageShowFrom.StartPosition = FormStartPosition.Manual;
                        m_listCCamerInfo[nCamID].m_objImageShowFrom.Location = new Point(390, 0);
                        m_listCCamerInfo[nCamID].m_objImageShowFrom.Show();
                    }
                    break;
                case 2:
                    {
                        m_listCCamerInfo[nCamID].m_objImageShowFrom.MdiParent = this;
                        m_listCCamerInfo[nCamID].m_objImageShowFrom.StartPosition = FormStartPosition.Manual;
                        m_listCCamerInfo[nCamID].m_objImageShowFrom.Location = new Point(0, 330);
                        m_listCCamerInfo[nCamID].m_objImageShowFrom.Show();
                    }
                    break;
                case 3:
                    {
                        m_listCCamerInfo[nCamID].m_objImageShowFrom.MdiParent = this;
                        m_listCCamerInfo[nCamID].m_objImageShowFrom.StartPosition = FormStartPosition.Manual;
                        m_listCCamerInfo[nCamID].m_objImageShowFrom.Location = new Point(390,330);
                        m_listCCamerInfo[nCamID].m_objImageShowFrom.Show();
                    }
                    break;
                default:
                    break;
            }

        }

        /// <summary>
        /// Select the corresponding device
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_cb_EnumDevice_SelectedIndexChanged(object sender, EventArgs e)
        {
            try
            {
                this.m_cb_EnumDevice.Focus();
                for (int i = 0; i < m_nCamNum; i++)
                {
                    if (m_cb_EnumDevice.Text == m_listCCamerInfo[i].m_strDisplayName)
                    {
                        m_nOperateID = i;
                        if (m_listCCamerInfo[m_nOperateID].m_bIsOpen)
                        {
                            __InitUI();
                        }
                        break;
                    }                    
                }
                // Update UI
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// Close device
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_btn_CloseDevice_Click(object sender, EventArgs e)
        {
            try
            {
                if (null != m_listCCamerInfo[m_nOperateID].m_objImageShowFrom)
                {
                    m_listCCamerInfo[m_nOperateID].m_objImageShowFrom.CloseDevice();
                    m_listCCamerInfo[m_nOperateID].m_bIsSnap = false;
                    m_listCCamerInfo[m_nOperateID].m_bIsOpen = false;
                }
                // Update UI
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// Start acquisition
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_btn_StartDevice_Click(object sender, EventArgs e)
        {
            try
            {
                m_listCCamerInfo[m_nOperateID].m_objImageShowFrom.StartDevice();
                m_listCCamerInfo[m_nOperateID].m_bIsSnap = true;
                // Update UI
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// Stop acquisition
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_btn_StopDevice_Click(object sender, EventArgs e)
        {
            try
            {
                m_listCCamerInfo[m_nOperateID].m_objImageShowFrom.StopDevice();
                m_listCCamerInfo[m_nOperateID].m_bIsSnap = false;
                
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// Release GxIAPINET
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void GxMultiCam_FormClosed(object sender, FormClosedEventArgs e)
        {
            try
            {
                this.m_cb_EnumDevice.Focus();
                __CloseAll();
                m_objIGXFactory.Uninit();

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void m_cb_AutoWhite_SelectedIndexChanged(object sender, EventArgs e)
        {
            try
            {
                if (!m_listCCamerInfo[m_nOperateID].m_bWhiteAutoSelectedIndex)
                {
                    return;
                }

                string strValue = m_cb_AutoWhite.Text;
                __SetEnumValue("BalanceWhiteAuto", strValue, m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl);
               
                m_listCCamerInfo[m_nOperateID].m_strBalanceWhiteAutoValue = strValue;

                // Update UI
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void __SetEnumValue(string strFeatureName, string strValue, IGXFeatureControl objIGXFeatureControl)
        {
            if (null != objIGXFeatureControl)
            {
                objIGXFeatureControl.GetEnumFeature(strFeatureName).SetValue(strValue);
            }
        }

        /// <summary>
        /// Limiting the acquisition speed level can only enter the integer
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_txt_AcqSpeedLevel_KeyPress(object sender, KeyPressEventArgs e)
        {
            if (!char.IsNumber(e.KeyChar) && e.KeyChar != 8)
            {
                e.Handled = true;
            }
        }

        /// <summary>
        /// Check whether the input value is decimals or integers
        /// </summary>
        /// <param name="sender">the object for control</param>
        /// <param name="e">KeyPressEvent</param>
        private void __CheckKeyPress(object sender, KeyPressEventArgs e)
        {
            if (!(((e.KeyChar >= '0') && (e.KeyChar <= '9')) || e.KeyChar <= 31))
            {
                if (e.KeyChar == '.')
                {
                    if (((TextBox)sender).Text.Trim().IndexOf('.') > -1)
                        e.Handled = true;
                }
                else
                    e.Handled = true;
            }
            else
            {
                if (e.KeyChar <= 31)
                {
                    e.Handled = false;
                }
                else if (((TextBox)sender).Text.Trim().IndexOf('.') > -1)
                {
                    if (((TextBox)sender).Text.Trim().Substring(((TextBox)sender).Text.Trim().IndexOf('.') + 1).Length >= 8)
                        e.Handled = true;
                }
            }
        }

        /// <summary>
        /// Check whether the enter key is pressed
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void GxMultiCam_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Return)
            {
                this.m_cb_EnumDevice.Focus();
            }
        }

        /// <summary>
        /// Check whether the input Exposure time is decimals or integers
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_txt_Shutter_KeyPress(object sender, KeyPressEventArgs e)
        {
            __CheckKeyPress(sender, e);
        }

        /// <summary>
        /// Check whether the input gain value is decimals or integers
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_txt_Gain_KeyPress(object sender, KeyPressEventArgs e)
        {
            __CheckKeyPress(sender, e);
        }

        /// <summary>
        ///  The response function indicate how to control AcquisitionSpeedLevel
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_txt_AcqSpeedLevel_Leave(object sender, EventArgs e)
        {
            Int64 nCurAcquisitionSpeedLevel = 0;   //AcquisitionSpeedLevel
            Int64 nMin = 0;                        //Min AcquisitionSpeedLevel
            Int64 nMax = 0;                        //Max AcquisitionSpeedLevel
            string strValue = "";                  
            try
            {
                if (!m_listCCamerInfo[m_nOperateID].m_bIsOpen)
                {
                    return;
                }

                strValue = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetIntFeature("AcquisitionSpeedLevel").GetValue().ToString();

                try
                {
                    nCurAcquisitionSpeedLevel = Convert.ToInt64(m_txt_AcqSpeedLevel.Text);    
                }
                catch (Exception)
                {
                    MessageBox.Show("Please input the correct AcquisitionSpeedLevel");
                    m_txt_AcqSpeedLevel.Text = strValue;
                    return;
                }

                if (null != m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl)
                {
                    if (!(m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.IsImplemented("AcquisitionSpeedLevel")))
                    {
                        m_listCCamerInfo[m_nOperateID].m_bAcqSpeedLevel = false;
                        return;
                    }

                    nMax = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetIntFeature("AcquisitionSpeedLevel").GetMax();
                    nMin = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetIntFeature("AcquisitionSpeedLevel").GetMin();

                    if (nCurAcquisitionSpeedLevel > nMax)
                    {
                        nCurAcquisitionSpeedLevel = nMax;
                    }

                    if (nCurAcquisitionSpeedLevel < nMin)
                    {
                        nCurAcquisitionSpeedLevel = nMin;
                    }

                    m_listCCamerInfo[m_nOperateID].m_bAcqSpeedLevel = true;
                    m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetIntFeature("AcquisitionSpeedLevel").SetValue(nCurAcquisitionSpeedLevel);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
            finally
            {
                if (null != m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl)
                {
                    if (!(m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.IsImplemented("AcquisitionSpeedLevel")))
                    {
                        m_listCCamerInfo[m_nOperateID].m_bAcqSpeedLevel = false;
                    }
                    else
                    {
                        m_listCCamerInfo[m_nOperateID].m_bAcqSpeedLevel = true;
                        m_txt_AcqSpeedLevel.Text = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetIntFeature("AcquisitionSpeedLevel").GetValue().ToString();
                    }
                }
                GxMultiCam.ActiveForm.Focus();
            }
        }

        /// <summary>
        ///  The response function indicate how to control exposure time
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_txt_Shutter_Leave(object sender, EventArgs e)
        {
            double dShutterValue = 0.0;              //Exposure time
            double dMin = 0.0;                       //Max exposure time
            double dMax = 0.0;                       //Min exposure time
            string strValue ="";                     
            try
            {
                if (!m_listCCamerInfo[m_nOperateID].m_bIsOpen)
                {
                    return;
                }

                strValue = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetFloatFeature("ExposureTime").GetValue().ToString();

                try
                {
                    dShutterValue = Convert.ToDouble(m_txt_Shutter.Text);
                }
                catch (Exception)
                {
                    MessageBox.Show("Please input the correct exposure time");
                    m_txt_Shutter.Text = strValue;
                    return;
                }

                //Get current device exposure time max and min value
                if (null != m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl)
                {
                    dMin = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetFloatFeature("ExposureTime").GetMin();
                    dMax = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetFloatFeature("ExposureTime").GetMax();

                    // Check whether the input value is within the exposure time range
                    if (dShutterValue > dMax)
                    {
                        dShutterValue = dMax;
                    }
                    if (dShutterValue < dMin)
                    {
                        dShutterValue = dMin;
                    }

                    m_txt_Shutter.Text = dShutterValue.ToString("F2");
                    m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetFloatFeature("ExposureTime").SetValue(dShutterValue);
                }
            }
            catch (CGalaxyException ex)
            {
                string strErrorInfo = "error code: " + ex.GetErrorCode().ToString() + "error information: " + ex.Message;
                MessageBox.Show(strErrorInfo);
            }
        }

        /// <summary>
        /// The response function indicate how to control gain value
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_txt_Gain_Leave(object sender, EventArgs e)
        {
            double dGain = 0;            //Gain value
            double dMin = 0.0;           //Min Gain value
            double dMax = 0.0;           //Max Gain value
            string strValue = "";                    
            try
            {
                if (!m_listCCamerInfo[m_nOperateID].m_bIsOpen)
                {
                    return;
                }
                strValue = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetFloatFeature("Gain").GetValue().ToString();
                try
                {
                    dGain = Convert.ToDouble(m_txt_Gain.Text);
                }
                catch (Exception)
                {
                    MessageBox.Show("Please input the correct gain value");
                    m_txt_Gain.Text = strValue; 
                    return;
                }

                //Get current device Gain max and min value
                if (null != m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl)
                {
                    dMin = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetFloatFeature("Gain").GetMin();
                    dMax = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetFloatFeature("Gain").GetMax();

                    // Check whether the input value is within the gain range
                    if (dGain > dMax)
                    {
                        dGain = dMax;
                    }

                    if (dGain < dMin)
                    {
                        dGain = dMin;
                    }

                    m_txt_Gain.Text = dGain.ToString("F2");
                    m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetFloatFeature("Gain").SetValue(dGain);
                }
            }
            catch (CGalaxyException ex)
            {
                string strErrorInfo = "error code: " + ex.GetErrorCode().ToString() + "Error information: " + ex.Message;
                MessageBox.Show(strErrorInfo);
            }
        }

        /// <summary>
        ///  display complete information in the combobox
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_cb_EnumDevice_DropDown(object sender, EventArgs e)
        {
            Graphics objGraphics = null;
            Font objFont = null;

            try
            {
                int nWidth = m_cb_EnumDevice.Width;
                objGraphics = m_cb_EnumDevice.CreateGraphics();
                objFont = m_cb_EnumDevice.Font;
                int nVertScrollBarWidth = (m_cb_EnumDevice.Items.Count > m_cb_EnumDevice.MaxDropDownItems) ? SystemInformation.VerticalScrollBarWidth : 0;
                int nNewWidth;
                for (int i = 0; i < m_cb_EnumDevice.Items.Count; i++)
                {
                    nNewWidth = (int)objGraphics.MeasureString(m_cb_EnumDevice.GetItemText(m_cb_EnumDevice.Items[i]), objFont).Width + nVertScrollBarWidth;
                    if (nWidth < nNewWidth)
                    {
                        nWidth = nNewWidth;
                    }
                }
                m_cb_EnumDevice.DropDownWidth = nWidth;
            }
            catch
            {

            }
            finally
            {
                if (null != objGraphics)
                {
                    objGraphics.Dispose();
                }
            }
        }

        /// <summary>
        /// Update white balance from once to off
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_timer_UpdateAutoWhite_Tick(object sender, EventArgs e)
        {
            try
            {
                string strValue = ""; //WhiteBalanceAuto
                Int32 i = 0;          

                // If the BalanceWhiteAuto mode is Once,After the successful setting, the actual BalanceWhiteAuto mode will be automatically turned off
                // So  once set to Once, the control is updated to off
                // Get the current BalanceWhiteAuto value from the device and then update to UI
                if (m_listCCamerInfo[m_nOperateID].m_strBalanceWhiteAutoValue == "Once")
                {
                    try
                    {
                        // Get current device white balance
                        if (null != m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl)
                        {
                            strValue = m_listCCamerInfo[m_nOperateID].m_objIGXFeatureControl.GetEnumFeature("BalanceWhiteAuto").GetValue();
                        }
                    }
                    catch (Exception)
                    {
                    }
                    m_listCCamerInfo[m_nOperateID].m_strBalanceWhiteAutoValue = strValue;

                    if (m_listCCamerInfo[m_nOperateID].m_strBalanceWhiteAutoValue == "Off")
                    {
                        for (i = 0; i < m_cb_AutoWhite.Items.Count; i++)
                        {
                            if (m_cb_AutoWhite.Items[i].ToString() == "Off")
                            {
                                m_cb_AutoWhite.SelectedIndex = i;
                                break;
                            }
                        }
                        __UpdateUI();
                    }
                }
            }
            catch (Exception)
            {
            }
        }
    }
}