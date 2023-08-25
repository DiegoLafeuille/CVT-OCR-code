using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Threading;
using System.IO;
using GxIAPINET;
using GxIAPINET.Sample.Common;


namespace GxSingleCam
{
    public partial class GxSingleCam : Form
    {
        bool                 m_bIsOpen                    = false;                           ///< The flag indicates whether the device has been opened.
        bool                 m_bIsSnap                    = false;                           ///< The flag indicates whether the device has been started acquisition.
        bool                 m_bTriggerMode               = false;                           ///< The flag indicates whether the device is supported trigger or not.
        bool                 m_bTriggerActive             = false;                           ///< The flag indicates whether the device is supported changing trigger activation or not.
        bool                 m_bTriggerSource             = false;                           ///< The flag indicates whether the device is supported selecting trigger source. or not.
        bool                 m_bWhiteAuto                 = false;                           ///< The flag indicates whether the device has been opened auto white balance or doesn't been.
        bool                 m_bBalanceRatioSelector      = false;                           ///< The flag indicates whether the device is supported selecting balance ratio selector.
        bool                 m_bWhiteAutoSelectedIndex    = true;                            ///< The flag indicates the conversion of white balance list box.
        IGXFactory           m_objIGXFactory              = null;                            ///< The object of class IGXFactory.
        IGXDevice            m_objIGXDevice               = null;                            ///< The object of class IGXDevice.
        IGXStream            m_objIGXStream               = null;                            ///< The object of class IGXStream.
        IGXFeatureControl    m_objIGXFeatureControl       = null;                            ///< The object of remote device feature control.
        IGXFeatureControl    m_objIGXStreamFeatureControl = null;                            ///< The object for stream feature control
        string               m_strBalanceWhiteAutoValue   = "Off";                           ///< The current value of BalanceWhiteAuto.
        GxBitmap             m_objGxBitmap                = null;                            ///< The object of class image display .
        string               m_strFilePath                = "";                              ///< The current path of executable.
       
        

        public GxSingleCam()
        {
            // Get the path of application currently executed  
            m_strFilePath = Directory.GetCurrentDirectory().ToString();
            m_strFilePath = m_strFilePath + "\\GxSingleCamImages";

            // Judging whether th path to exist or doesn't
            if (!Directory.Exists(m_strFilePath))
            {
                Directory.CreateDirectory(m_strFilePath);
            }

            InitializeComponent();
        }

        /// <summary>
        /// Init UI After the device has been opened.
        /// </summary>
        private void __InitUI()
        {
            __InitEnumComBoxUI(m_cb_TriggerMode, "TriggerMode", m_objIGXFeatureControl, ref m_bTriggerMode);                      //Init TriggerMode Widget
            __InitEnumComBoxUI(m_cb_TriggerSource, "TriggerSource", m_objIGXFeatureControl, ref m_bTriggerSource);                //Init TriggerSource Widget
            __InitEnumComBoxUI(m_cb_TriggerActivation, "TriggerActivation", m_objIGXFeatureControl, ref m_bTriggerActive);        //Init TriggerActivation Widget
            __InitShutterUI();                                                                                                    //Init ExposureTime Widget
            __InitGainUI();                                                                                                       //Init Gain Widget
            __InitWhiteRatioUI();                                                                                                 //Init Widgets  related to  BalanceWhite Ratio 
            __InitEnumComBoxUI(m_cb_AutoWhite, "BalanceWhiteAuto", m_objIGXFeatureControl, ref m_bWhiteAuto);                     //Init BalanceWhiteAuto Widget
            __InitEnumComBoxUI(m_cb_RatioSelector, "BalanceRatioSelector", m_objIGXFeatureControl, ref m_bBalanceRatioSelector);  //Init BalanceRatioSelector Widget


            //Get the current value of BalanceWhiteAuto
            bool bIsImplemented = false;             //The flag indicates whether is implementary or not.
            bool bIsReadable = false;                //The flag indicated whether is available to read or not.

            // Check whether the device supports BalanceWhiteAuto
            if (null != m_objIGXFeatureControl)
            {
                bIsImplemented = m_objIGXFeatureControl.IsImplemented("BalanceWhiteAuto");
                bIsReadable = m_objIGXFeatureControl.IsReadable("BalanceWhiteAuto");
                if (bIsImplemented)
                {
                    if (bIsReadable)
                    {
                        //Get the current value of BalanceWhiteAuto
                        m_strBalanceWhiteAutoValue = m_objIGXFeatureControl.GetEnumFeature("BalanceWhiteAuto").GetValue();
                    }
                }
            }
        }


        /// <summary>
        /// According to the function name, set value for the enumeration-type variables.
        /// </summary>
        /// <param name="strFeatureName">enumeration-type function name</param>
        /// <param name="strValue">value of the function name</param>
        /// <param name="objIGXFeatureControl">Object of the device feature control class</param>
        private void __SetEnumValue(string strFeatureName, string strValue, IGXFeatureControl objIGXFeatureControl)
        {
            if (null != objIGXFeatureControl)
            {
                //set current value
                objIGXFeatureControl.GetEnumFeature(strFeatureName).SetValue(strValue);
            }
        }

        /// <summary>
        /// Init UI Widget of enumeration-type function
        /// </summary>
        /// <param name="cbEnum">ComboBox Widget Name</param>
        /// <param name="strFeatureName">enumeration-type function name</param>
        /// <param name="objIGXFeatureControl">Object of the device feature control class</param>
        /// <param name="bIsImplemented">Whether is implementary or not</param>
        private void __InitEnumComBoxUI(ComboBox cbEnum, string strFeatureName, IGXFeatureControl objIGXFeatureControl, ref bool bIsImplemented)
        {
            string strTriggerValue = "";                   //The current selector
            List<string> list = new List<string>();        //Combox will be filled by this list.
            bool bIsReadable = false;                      //Whether is implementary or not.

            // Check whether the device supports this enumeration-type function
            if (null != objIGXFeatureControl)
            {

                bIsImplemented = objIGXFeatureControl.IsImplemented(strFeatureName);

                // It will be returned, if don't implement this enumeration-type function.
                if (!bIsImplemented)
                {
                    return;
                }

                bIsReadable = objIGXFeatureControl.IsReadable(strFeatureName);

                if (bIsReadable)
                {
                    list.AddRange(objIGXFeatureControl.GetEnumFeature(strFeatureName).GetEnumEntryList());
                    //Get current value of the function.
                    strTriggerValue = objIGXFeatureControl.GetEnumFeature(strFeatureName).GetValue();
                }

            }

            //Clear ComBox and update data to the window.
            cbEnum.Items.Clear();
            foreach (string str in list)
            {
                cbEnum.Items.Add(str);
            }

            //Comparing the value from camera with the enumerative value,  to obtain the index and update UI widget.
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
        /// Init Widgets  related to  BalanceWhite Ratio 
        /// </summary>
        private void __InitWhiteRatioUI()
        {
            double dWhiteRatio = 0.0;                       //Current Value of ExposureTime
            double dMin = 0.0;                              //Min Value
            double dMax = 0.0;                              //Max Value
            string strUnit = "";                            //Unit
            string strText = "";                            //The content of display
            bool bIsBalanceRatio = false;                   //The flag indicates whether  the device is supported balance ratio
            
            //Get the value of white balance ratio  and it's min value, max value, unit.
            if (null != m_objIGXFeatureControl)
            {
                bIsBalanceRatio = m_objIGXFeatureControl.IsImplemented("BalanceRatio");
                if (!bIsBalanceRatio)
                {
                    m_txt_BalanceRatio.Enabled = false; ;
                    return;
                }
                dWhiteRatio = m_objIGXFeatureControl.GetFloatFeature("BalanceRatio").GetValue();
                dMin = m_objIGXFeatureControl.GetFloatFeature("BalanceRatio").GetMin();
                dMax = m_objIGXFeatureControl.GetFloatFeature("BalanceRatio").GetMax();
                strUnit = m_objIGXFeatureControl.GetFloatFeature("BalanceRatio").GetUnit();
            }

            //Update the range of white balance and unit to UI widget.
            strText = string.Format("Whitebalance Ratio({0}~{1}){2}", dMin.ToString("0.00"), dMax.ToString("0.00"), strUnit);
            m_lbl_WhiteRatio.Text = strText;

            //Update the value of white balance ratio to UI.
            m_txt_BalanceRatio.Text = dWhiteRatio.ToString("0.00");
        }


        /// <summary>
        /// Init ExposureTime Widget
        /// </summary>
        private void __InitShutterUI()
        {
            double dCurShuter = 0.0;                       //The current value of exposure time
            double dMin = 0.0;                             //Min
            double dMax = 0.0;                             //Max
            string strUnit = "";                           //Unit
            string strText = "";                           //The content of display

            //Get the value of exposure time  and it's min value, max value, unit.
            if (null != m_objIGXFeatureControl)
            {
                dCurShuter = m_objIGXFeatureControl.GetFloatFeature("ExposureTime").GetValue();
                dMin = m_objIGXFeatureControl.GetFloatFeature("ExposureTime").GetMin();
                dMax = m_objIGXFeatureControl.GetFloatFeature("ExposureTime").GetMax();
                strUnit = m_objIGXFeatureControl.GetFloatFeature("ExposureTime").GetUnit();
            }

            //Update the range of exposure time and unit to UI widget.
            strText = string.Format("ExposureTime({0}~{1}){2}", dMin.ToString("0.00"), dMax.ToString("0.00"), strUnit);
            m_lbl_Shutter.Text = strText;

            //Update the value of exposure time to UI.
            m_txt_Shutter.Text = dCurShuter.ToString("0.00");
        }

        /// <summary>
        /// Init Gain Widget
        /// </summary>
        private void __InitGainUI()
        {
            double dCurGain = 0;             //The current value of gain
            double dMin = 0.0;               //Min
            double dMax = 0.0;               //Max
            string strUnit = "";             //Unit
            string strText = "";            //The content of display

            //Get the value of gain  and it's min value, max value, unit.
            if (null != m_objIGXFeatureControl)
            {
                dCurGain = m_objIGXFeatureControl.GetFloatFeature("Gain").GetValue();
                dMin = m_objIGXFeatureControl.GetFloatFeature("Gain").GetMin();
                dMax = m_objIGXFeatureControl.GetFloatFeature("Gain").GetMax();
                strUnit = m_objIGXFeatureControl.GetFloatFeature("Gain").GetUnit();
            }

            //Update the range of gain and unit to UI widget.
            strText = string.Format("Gain({0}~{1}){2}", dMin.ToString("0.00"), dMax.ToString("0.00"), strUnit);
            m_lbl_Gain.Text = strText;

            //Update the value of gain to UI.
            string strCurGain = dCurGain.ToString("0.00");
            m_txt_Gain.Text = strCurGain;
        }

        /// <summary>
        /// Update the enable status of UI Widget
        /// </summary>
        void __UpdateUI()
        {
            //The enable status related camera control 
            m_btn_OpenDevice.Enabled = !m_bIsOpen;
            m_btn_CloseDevice.Enabled = m_bIsOpen;
            m_btn_StartDevice.Enabled = m_bIsOpen && !m_bIsSnap;
            m_btn_StopDevice.Enabled = m_bIsSnap;

            //The enable status related camera parameter
            m_cb_TriggerMode.Enabled = m_bIsOpen && m_bTriggerMode;
            m_cb_TriggerSource.Enabled = m_bIsOpen && m_bTriggerSource;
            m_cb_TriggerActivation.Enabled = m_bIsOpen && m_bTriggerActive;
            m_cb_RatioSelector.Enabled = m_bIsOpen && m_bBalanceRatioSelector;


            m_txt_Shutter.Enabled = m_bIsOpen;
            m_txt_Gain.Enabled = m_bIsOpen;
            m_txt_BalanceRatio.Enabled = m_bIsOpen
                                         && m_bBalanceRatioSelector
                                         && (m_strBalanceWhiteAutoValue == "Off");


            m_cb_AutoWhite.Enabled = m_bIsOpen && m_bWhiteAuto;
            m_btn_SoftTriggerCommand.Enabled = m_bIsOpen;
        }

        /// <summary>
        /// Init camera 
        /// </summary>
        void __InitDevice()
        {
            if (null != m_objIGXFeatureControl)
            {
                //Setting the acquisition mode is continuous.
                m_objIGXFeatureControl.GetEnumFeature("AcquisitionMode").SetValue("Continuous");
            }
        }

        /// <summary>
        /// Close stream
        /// </summary>
        private void __CloseStream()
        {
            try
            {
                //Close stream
                if (null != m_objIGXStream)
                {
                    m_objIGXStream.Close();
                    m_objIGXStream = null;
                    m_objIGXStreamFeatureControl = null;
                }
            }
            catch (Exception)
            {
            }
        }

        /// <summary>
        /// Close device
        /// </summary>
        private void __CloseDevice()
        {
            try
            {
                //Close device
                if (null != m_objIGXDevice)
                {
                    m_objIGXDevice.Close();
                    m_objIGXDevice = null;
                }
            }
            catch (Exception)
            {
            }
        }

        /// <summary> 
        /// Loading windows, initialize UI and library.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void GxSingleCam_Load(object sender, EventArgs e)
        {
            try
            {
                //Update UI
                __UpdateUI();

                m_objIGXFactory = IGXFactory.GetInstance();
                m_objIGXFactory.Init();

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            } 
        }

        /// <summary>
        /// Open device and stream
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_btn_OpenDevice_Click(object sender, EventArgs e)
        {
            try
            {
                List<IGXDeviceInfo> listGXDeviceInfo = new List<IGXDeviceInfo>();

                //close stream
                __CloseStream();

                // If the device has been opened, it will be closed in order to the device can be opend again after initializing failed.
                __CloseDevice();

                m_objIGXFactory.UpdateDeviceList(200, listGXDeviceInfo);

                // Check it has found device or not.
                if (listGXDeviceInfo.Count <= 0)
                {
                    MessageBox.Show("No device has been found!");
                    return;
                }

                // If the device has been opened then close it to ensure the camera can be opened again.
                if (null != m_objIGXDevice)
                {
                    m_objIGXDevice.Close();
                    m_objIGXDevice = null;
                }

                //Open the first device in the device list has been found
                m_objIGXDevice = m_objIGXFactory.OpenDeviceBySN(listGXDeviceInfo[0].GetSN(), GX_ACCESS_MODE.GX_ACCESS_EXCLUSIVE);
                m_objIGXFeatureControl = m_objIGXDevice.GetRemoteFeatureControl();


                //Open the stream.
                if (null != m_objIGXDevice)
                {
                    m_objIGXStream = m_objIGXDevice.OpenStream(0);
                    m_objIGXStreamFeatureControl = m_objIGXStream.GetFeatureControl();
                }

                // It is recommended that the user set the camera's stream channel packet length value
                // according to the current network environment after turning on 
                // the network camera to improve the collection performance of the network camera. 
                // For the setting method, refer to the following code.
                GX_DEVICE_CLASS_LIST objDeviceClass = m_objIGXDevice.GetDeviceInfo().GetDeviceClass();
                if (GX_DEVICE_CLASS_LIST.GX_DEVICE_CLASS_GEV == objDeviceClass)
                {
                    // Determine whether the device supports the stream channel packet function.
                    if (true == m_objIGXFeatureControl.IsImplemented("GevSCPSPacketSize"))
                    {
                        // Get the optimal packet length value of the current network environment
                        uint nPacketSize = m_objIGXStream.GetOptimalPacketSize();
                        // Set the optimal packet length value to the stream channel packet length of the current device.
                        m_objIGXFeatureControl.GetIntFeature("GevSCPSPacketSize").SetValue(nPacketSize);
                    }
                }

                //Initialize the device parameters.
                __InitDevice();

                // Initialize UI widget
                __InitUI();

                m_objGxBitmap = new GxBitmap(m_objIGXDevice, m_pic_ShowImage);

                // Update the flag that indicates whether the device has been opened.
                m_bIsOpen = true;

                // Update the enable status of UI Widget
                __UpdateUI();

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            } 
        }

        /// <summary>
        /// Close stream and close device
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_btn_CloseDevice_Click(object sender, EventArgs e)
        {
            try
            {
                try
                {
                    // If the device doesn't stop to capture, it will be stopped to capture at first.
                    if (m_bIsSnap)
                    {
                        if (null != m_objIGXFeatureControl)
                        {
                            m_objIGXFeatureControl.GetCommandFeature("AcquisitionStop").Execute();
                            m_objIGXFeatureControl = null;
                        }
                    }
                }
                catch (Exception)
                {

                }

                m_bIsSnap = false;

                try
                {
                    //Close the acquisition of stream ,unregister the  capture callback function and close stream.
                    if (null != m_objIGXStream)
                    {
                        m_objIGXStream.StopGrab();

                        //Unregister the  capture callback function
                        m_objIGXStream.UnregisterCaptureCallback();
                        m_objIGXStream.Close();
                        m_objIGXStream = null;
                        m_objIGXStreamFeatureControl = null;
                    }
                }
                catch (Exception)
                {

                }

                try
                {
                    //Close the device
                    if (null != m_objIGXDevice)
                    {
                        m_objIGXDevice.Close();
                        m_objIGXDevice = null;
                    }
                }
                catch (Exception)
                {

                }

                m_bIsOpen = false;

                //Update the enable status of UI Widget
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            } 
        }

        /// <summary>
        /// Start to capture
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_btn_StartDevice_Click(object sender, EventArgs e)
        {
            try
            {
                if (null != m_objIGXStreamFeatureControl)
                {
                    //Set StreamBufferHandlingMode
                    m_objIGXStreamFeatureControl.GetEnumFeature("StreamBufferHandlingMode").SetValue("OldestFirst");
                }

                //Open the acquisition of stream.
                if (null != m_objIGXStream)
                {
                    //The first parameter of the function RegisterCaptureCallback is belong to custom defined parameter(it's type must be quotative type).
                    //If want to use the parameter, the user can used it in the delegate function.
                    m_objIGXStream.RegisterCaptureCallback(this, __CaptureCallbackPro);
                    m_objIGXStream.StartGrab();
                }

                // Send the command of starting acquisition.
                if (null != m_objIGXFeatureControl)
                {
                    m_objIGXFeatureControl.GetCommandFeature("AcquisitionStart").Execute();
                }
                m_bIsSnap = true;

                // Update the enable status of UI Widget
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            } 
        }

        /// <summary>
        /// Stop to capture
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_btn_StopDevice_Click(object sender, EventArgs e)
        {
            try
            {

                //Send the command of stopping acquisition.
                if (null != m_objIGXFeatureControl)
                {
                    m_objIGXFeatureControl.GetCommandFeature("AcquisitionStop").Execute();
                }


                //Close the stream
                if (null != m_objIGXStream)
                {
                    m_objIGXStream.StopGrab();

                    //Unregister the capture callback
                    m_objIGXStream.UnregisterCaptureCallback();
                }


                m_bIsSnap = false;

                // Update the enable status of UI Widget
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            } 
        }

        /// <summary>
        /// The display and storage of images
        /// </summary>
        /// <param name="objIFrameData">The object of the struct of IFrameData</param>
        void ImageShowAndSave(IFrameData objIFrameData)
        {
            try
            {
                m_objGxBitmap.Show(objIFrameData);
            }
            catch (Exception)
            {
            }

            // Check whether is need to save image or not.
            if (m_bSaveBmpImg.Checked)
            {
                DateTime dtNow = System.DateTime.Now;  // Get the current system time
                string strDateTime = dtNow.Year.ToString() + "_"
                                   + dtNow.Month.ToString() + "_"
                                   + dtNow.Day.ToString() + "_"
                                   + dtNow.Hour.ToString() + "_"
                                   + dtNow.Minute.ToString() + "_"
                                   + dtNow.Second.ToString() + "_"
                                   + dtNow.Millisecond.ToString();

                string stfFileName = m_strFilePath + "\\" + strDateTime + ".bmp";  // The default name of image will be saved
                m_objGxBitmap.SaveBmp(objIFrameData, stfFileName);
            }
        }

        /// <summary>
        /// The capturing callback function, used to get the information of the image cand to display
        /// </summary>
        /// <param name="obj">the custom defined parameter</param>
        /// <param name="objIFrameData">the object of the struct of image information </param>
        private void __CaptureCallbackPro(object objUserParam, IFrameData objIFrameData)
        {
            try
            {
                GxSingleCam objGxSingleCam = objUserParam as GxSingleCam;
                objGxSingleCam.ImageShowAndSave(objIFrameData);
            }
            catch (Exception)
            {
            }
        }

        /// <summary>
        /// The response function of the combox that used to switch the trigger mode.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_cb_TriggerMode_SelectedIndexChanged(object sender, EventArgs e)
        {
            try
            {
                string strValue = m_cb_TriggerMode.Text;
                __SetEnumValue("TriggerMode", strValue, m_objIGXFeatureControl);

                // Update the enable status of UI Widget
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            } 
        }

        /// <summary>
        /// The response function of the combox that used to switch the trigger source.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_cb_TriggerSource_SelectedIndexChanged(object sender, EventArgs e)
        {
            try
            {
                string strValue = m_cb_TriggerSource.Text;
                __SetEnumValue("TriggerSource", strValue, m_objIGXFeatureControl);

                // Update the enable status of UI Widget
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            } 
        }

        /// <summary>
        /// The response function of the combox that used to switch the trigger activation.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_cb_TriggerActivation_SelectedIndexChanged(object sender, EventArgs e)
        {
            try
            {
                string strValue = m_cb_TriggerActivation.Text;
                __SetEnumValue("TriggerActivation", strValue, m_objIGXFeatureControl);

                // Update the enable status of UI Widget
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            } 
        }


        /// <summary>
        /// Send soft trigger command
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_btn_SoftTriggerCommand_Click(object sender, EventArgs e)
        {
            try
            {
                //Send soft trigger command
                if (null != m_objIGXFeatureControl)
                {
                    m_objIGXFeatureControl.GetCommandFeature("TriggerSoftware").Execute();
                }

                // Update the enable status of UI Widget
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            } 
        }


        /// <summary>
        /*-----------------------------------------------------------------------
        * When the value of auto white balance switches from the other two value to once,
        * it will became off  automatically by the device itself after setting the value of once.
        * In order to make the UI widget status is consistent with the device, will simulate
        * the change process with program.The change process is checking the value of auto white balance 
        * whether is setted once, if it is true, Updating the auto white balance widget 
        * to off immediately(by UpdateWhiteAutoUI() fuction). But in the process, 
        * the function of m_cb_AutoWhite_SelectedIndexChanged() will be executed twice,
        * and the value of the device auto white balance has became off, and if setting the value
        * of auto white balance again, the process of setting once may be interrupted 
        * causing the white balance doesn't work. Parameter m_bWhiteAutoSelectedIndex 
        * is introduced in order to solve the problem of function reentrant.
        ------------------------------------------------------------------------*/
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_cb_AutoWhite_SelectedIndexChanged(object sender, EventArgs e)
        {
            try
            {
                if (!m_bWhiteAutoSelectedIndex)
                {
                    return;
                }
                string strValue = m_cb_AutoWhite.Text;
                __SetEnumValue("BalanceWhiteAuto", strValue, m_objIGXFeatureControl);
                m_strBalanceWhiteAutoValue = strValue;

                // Get the value of white balance to update UI widget.
                __InitWhiteRatioUI();

                // Update the enable status of UI Widget
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            } 
        }

        /// <summary>
        /// The response function for the combox that used to switch the white balance selector.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_cb_RatioSelector_SelectedIndexChanged(object sender, EventArgs e)
        {
            try
            {
                string strValue = m_cb_RatioSelector.Text;
                __SetEnumValue("BalanceRatioSelector", strValue, m_objIGXFeatureControl);

                // Get the value of white balance to update UI widget.
                __InitWhiteRatioUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            } 
        }

        /// <summary>
        /// Close the stream, close the device and execute the uninitialized function of GXFactory object.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void GxSingleCam_FormClosed(object sender, FormClosedEventArgs e)
        {
            try
            {
                // If the device doesn't stop to capture, it will be stopped to capture at first.
                if (m_bIsSnap)
                {
                    if (null != m_objIGXFeatureControl)
                    {
                        m_objIGXFeatureControl.GetCommandFeature("AcquisitionStop").Execute();
                    }
                }
            }
            catch (Exception)
            {

            }

            try
            {
                //Stop the acquisition of stream, unregister the capture callback and close stream.
                if (null != m_objIGXStream)
                {
                    m_objIGXStream.StopGrab();
                    m_objIGXStream.UnregisterCaptureCallback();
                    m_objIGXStream.Close();
                    m_objIGXStream = null;
                    m_objIGXStreamFeatureControl = null;
                }
            }
            catch (Exception)
            {

            }

            try
            {
                //Close device
                if (null != m_objIGXDevice)
                {
                    m_objIGXDevice.Close();
                    m_objIGXDevice = null;
                }
            }
            catch (Exception)
            {

            }
           
            try
            {
                //Execute the uninitialized function of GXFactory object
                if (null != m_objIGXFactory)
                {
                    m_objIGXFactory.Uninit();
                }
            }
            catch (Exception)
            {

            }
        }

        /// <summary>
        /// Check the input value whether is decimals or integer, and the demands of decimals is less than eight places after decimal point.
        /// </summary>
        /// <param name="sender">the object of UI widget</param>
        /// <param name="e"> The response function of keyboard</param>
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
        /// Check the input value of exposure time whether is decimals or integer.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_txt_Shutter_KeyPress(object sender, KeyPressEventArgs e)
        {
            __CheckKeyPress(sender, e);
        }

        /// <summary>
        /// Check the input value of gain whether is decimals or integer.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_txt_Gain_KeyPress(object sender, KeyPressEventArgs e)
        {
            __CheckKeyPress(sender, e);
        }

        /// <summary>
        /// Check the input value of white balance ratio whether is decimals or integer.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_txt_BalanceRatio_KeyPress(object sender, KeyPressEventArgs e)
        {
            __CheckKeyPress(sender, e);
        }

        /// <summary>
        /// Check whether the enter key was pressed.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void GxSingleCam_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Return)
            {
                this.m_pic_ShowImage.Focus();
            }
        }

        /// <summary>
        /// The response function of the exposure time edit widget lost focus.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_txt_Shutter_Leave(object sender, EventArgs e)
        {
            double dShutterValue = 0.0;              //Exposure time value
            double dMin = 0.0;                       //Min value of exposure time
            double dMax = 0.0;                       //max value of exposure time

            try
            {

                if (!m_bIsOpen)
                {
                    return;
                }

                try
                {
                    dShutterValue = Convert.ToDouble(m_txt_Shutter.Text);
                }
                catch (Exception)
                {
                    __InitShutterUI();
                    MessageBox.Show("Please input the correct value of exposure time.");
                    return;
                }

                //Get the current value and min value and max value of exposure time, and the unit.
                if (null != m_objIGXFeatureControl)
                {
                    dMin = m_objIGXFeatureControl.GetFloatFeature("ExposureTime").GetMin();
                    dMax = m_objIGXFeatureControl.GetFloatFeature("ExposureTime").GetMax();

                    //Check the input value whether is in the range of exposure time or not.
                    //If the input value is bigger than the max value, it will be set the max value.
                    if (dShutterValue > dMax)
                    {
                        dShutterValue = dMax;
                    }
                    //If the input value is smaller than the max value, it will be set the min value.
                    if (dShutterValue < dMin)
                    {
                        dShutterValue = dMin;
                    }

                    m_txt_Shutter.Text = dShutterValue.ToString("F2");
                    m_objIGXFeatureControl.GetFloatFeature("ExposureTime").SetValue(dShutterValue);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            } 
        }

        /// <summary>
        /// The response function of the gain edit widget lost focus.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_txt_Gain_Leave(object sender, EventArgs e)
        {
            double dGain = 0;            //the current value of gain
            double dMin = 0.0;           //Miin value
            double dMax = 0.0;           //Max value

            try
            {
                if (!m_bIsOpen)
                {
                    return;
                }

                try
                {
                    dGain = Convert.ToDouble(m_txt_Gain.Text);
                }
                catch (Exception)
                {
                    __InitGainUI();
                    MessageBox.Show("Please input the correct value of gain");
                    return;
                }

                //Get the current value and min value and max value of gain, and the unit.
                if (null != m_objIGXFeatureControl)
                {
                    dMin = m_objIGXFeatureControl.GetFloatFeature("Gain").GetMin();
                    dMax = m_objIGXFeatureControl.GetFloatFeature("Gain").GetMax();

                    //Check the input value whether is in the range of gain or not.
                    //If the input value is bigger than the max value, it will be set the max value.
                    if (dGain > dMax)
                    {
                        dGain = dMax;
                    }

                    //If the input value is smaller than the max value, it will be set the min value.
                    if (dGain < dMin)
                    {
                        dGain = dMin;
                    }

                    m_txt_Gain.Text = dGain.ToString("F2");
                    m_objIGXFeatureControl.GetFloatFeature("Gain").SetValue(dGain);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            } 
        }

        /// <summary>
        /// The response function of the white balance ratio edit widget lost focus.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_txt_BalanceRatio_Leave(object sender, EventArgs e)
        {
            double dBalanceRatio = 0;    //the current value of white balance ratio
            double dMin = 0.0;           //Min value
            double dMax = 0.0;           //Max value
            

            try
            {
                if (!m_bIsOpen)
                {
                    return;
                }

                try
                {
                    dBalanceRatio = Convert.ToDouble(m_txt_BalanceRatio.Text);
                }
                catch (Exception)
                {
                    __InitWhiteRatioUI();
                    MessageBox.Show("Please input the correct value of white balance ratio");
                    return;
                }

                //Get the current value and min value and max value of gain, and the unit.
                if (null != m_objIGXFeatureControl)
                {
                    dMin = m_objIGXFeatureControl.GetFloatFeature("BalanceRatio").GetMin();
                    dMax = m_objIGXFeatureControl.GetFloatFeature("BalanceRatio").GetMax();

                    //Check the input value whether is in the range of white balance ratio or not.
                    //If the input value is bigger than the max value, it will be set the max value.
                    if (dBalanceRatio > dMax)
                    {
                        dBalanceRatio = dMax;
                    }

                    //If the input value is smaller than the max value, it will be set the min value.
                    if (dBalanceRatio < dMin)
                    {
                        dBalanceRatio = dMin;
                    }

                    m_txt_BalanceRatio.Text = dBalanceRatio.ToString("F2");
                    m_objIGXFeatureControl.GetFloatFeature("BalanceRatio").SetValue(dBalanceRatio);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            } 
        }

        /// <summary>
        /// Update the value of auto white balance from once to off.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_timer_UpdateAutoWhite_Tick(object sender, EventArgs e)
        {
            try
            {
                string strValue = ""; //the value of auto white balance
                Int32 i = 0;          //Loop variables


                // If the value of auto white balance is once will be setted , the actual value of auto white balance will become off automatically after setting once successfuly.
                // In order to the UI widget is consistent with the device, will simulate the change process with program 
                // that is Updating the auto white balance widget to off immediately after setting the value of once for device.
                if (m_strBalanceWhiteAutoValue == "Once")
                {
                    try
                    {
                        //Get the enumerated values of auto white balance.
                        if (null != m_objIGXFeatureControl)
                        {
                            strValue = m_objIGXFeatureControl.GetEnumFeature("BalanceWhiteAuto").GetValue();
                        }
                    }
                    catch (Exception)
                    {
                    }
                    m_strBalanceWhiteAutoValue = strValue;

                    if (m_strBalanceWhiteAutoValue == "Off")
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
