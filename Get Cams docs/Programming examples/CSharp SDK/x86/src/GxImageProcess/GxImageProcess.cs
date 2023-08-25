using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Threading;
using GxIAPINET;
using GxIAPINET.Sample.Common;

namespace GxImageProcess
{
    public partial class GxImageProcess : Form
    {
        bool                                    m_bIsOpen = false;                                            ///< The flag indicates whether the device has been opened. 
        bool                                    m_bIsSnap = false;                                            ///< The flag indicates whether the device has been started acquisition.
        bool                                    m_bColorFilter = false;                                       ///< The flag indicates whether the device is supported color filter or not.
        bool                                    m_bAwbLampHouse = false;                                      ///< The flag indicates whether the device supports AwbLampHouse.
        bool                                    m_bWhiteAutoSelectedIndex = true;                             ///< The flag indicates the conversion of white balance list box.
        IGXFactory                              m_objIGXFactory = null;                                       ///< The object of class IGXFactory.
        IGXDevice                               m_objIGXDevice              = null;                           ///< The object of class IGXDevice
        IGXStream                               m_objIGXStream              = null;                           ///< The object of class IGXStream
        IGXFeatureControl                       m_objIGXFeatureControl      = null;                           ///< The object of remote device feature control.
        IGXFeatureControl                       m_objIGXStreamFeatureControl = null;                          ///< The object for stream feature control
        IImageProcessConfig                     m_objCfg = null;                                              ///< The object of class IImageProcessConfig
        GxBitmap                                m_objGxBitmap               = null;                           ///< The object of class GxBitmap
        string                                  m_strPixelColorFilter       = null;                           ///< The string of pixel color filter
        string                                  m_strBalanceWhiteAutoValue  = "Off";                          ///< Current value of BalanceWhiteAuto.
        bool                                    m_bEnableColorCorrect       = false;                          ///<Color correction enables flag bit
        bool                                    m_bEnableGamma              = false;                          ///<Gamma enabled flag bit
        bool                                    m_bEnableSharpness          = false;                          ///<Sharpen enable flag bit 
        bool                                    m_bEnableAutoWhite          = false;                          ///<Automatic white balance enable flag bit
        bool                                    m_bEnableAwbLight           = false;                          ///<Automatic white balance light source enable flag bit
        bool                                    m_bEnableDenoise            = false;                          ///<Image denoising enable flag bit
        bool                                    m_bEnableSaturation         = false;                          ///<Saturation enable flag bit
        bool                                    m_bEnumDevices              = false;                          ///<Whether to enumerate to the device flag bit
        List<IGXDeviceInfo>                     m_listGXDeviceInfo;                                           ///<A container for holding enumerated devices

        public GxImageProcess()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Init UI After the device has been opened.
        /// </summary>
        private void __InitUI()
        {

            if (null != m_objCfg)
            {
                if (m_bColorFilter)
                {
                    __InitEnumComBoxUI(m_cb_AutoWhite, "BalanceWhiteAuto", m_objIGXFeatureControl, ref m_bColorFilter);                     //Initialization of automatic white balance
                    __InitEnumComBoxUI(m_cb_AwbLight, "AWBLampHouse", m_objIGXFeatureControl, ref m_bAwbLampHouse);                         //White balance channel selection

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

                    m_ScrollBar_Saturation.Value = m_objCfg.GetSaturationParam();
                    m_txt_Saturation.Text = m_objCfg.GetSaturationParam().ToString();
                    m_checkBox_OpenColorCorrect.Checked = (m_objCfg.IsColorCorrection() && !m_bEnableColorCorrect);
                    m_checkBox_OpenDenoise.Checked = m_objCfg.IsDenoise();

                    m_bEnableAutoWhite = false;
                    m_bEnableAwbLight = false;
                    m_bEnableColorCorrect = false;
                    m_bEnableDenoise = false;
                    m_bEnableSaturation = false;
                }
                else
                {                 
                    m_bEnableAutoWhite = true;
                    m_bEnableAwbLight = true;
                    m_bEnableColorCorrect = true;
                    m_bEnableDenoise = true;
                    m_bEnableSaturation = true;

                }

                // Initialize the location of the slider
                m_ScrollBar_Sharpen.Value = Convert.ToInt32(m_objCfg.GetSharpenParam() * 10);
                m_ScrollBar_Gamma.Value =Convert.ToInt32( m_objCfg.GetGammaParam() * 10);
                m_ScrollBar_Contrast.Value = m_objCfg.GetContrastParam();
                m_ScrollBar_Lightness.Value = m_objCfg.GetLightnessParam();

                // Initialize the value of edit widget
                m_txt_Sharpen.Text = m_objCfg.GetSharpenParam().ToString();
                m_txt_Gamma.Text = m_objCfg.GetGammaParam().ToString();
                m_txt_Contrast.Text = m_objCfg.GetContrastParam().ToString();
                m_txt_Lightness.Text = m_objCfg.GetLightnessParam().ToString();

                // Initialize the status of check box widget
                m_CheckBox__BadPixelCorrect.Checked = m_objCfg.IsDefectivePixelCorrect();
                m_checkBox_OpenSharpen.Checked = m_objCfg.IsSharpen();
                m_checkBox_Accelerate.Checked = m_objCfg.IsAccelerate();

            }
   
        }

        /// <summary>
        /// Get the best bit by GX_PIXEL_FORMAT_ENTRY
        /// </summary>
        /// <param name="em">Pixel format</param>
        /// <returns>The best bit</returns>
        private GX_VALID_BIT_LIST __GetBestValudBit(GX_PIXEL_FORMAT_ENTRY emPixelFormatEntry)
        {
            GX_VALID_BIT_LIST emValidBits = GX_VALID_BIT_LIST.GX_BIT_0_7;
            switch (emPixelFormatEntry)
            {
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_MONO8:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GR8:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_RG8:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GB8:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_BG8:
                    {
                        emValidBits = GX_VALID_BIT_LIST.GX_BIT_0_7;
                        break;
                    }
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_MONO10:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GR10:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_RG10:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GB10:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_BG10:
                    {
                        emValidBits = GX_VALID_BIT_LIST.GX_BIT_2_9;
                        break;
                    }
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_MONO12:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GR12:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_RG12:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GB12:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_BG12:
                    {
                        emValidBits = GX_VALID_BIT_LIST.GX_BIT_4_11;
                        break;
                    }
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_MONO14:
                    {
                        //So far, there is no such data format
                        break;
                    }
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_MONO16:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GR16:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_RG16:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GB16:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_BG16:
                    {
                        //So far, there is no such data format
                        break;
                    }
                default:
                    break;
            }
            return emValidBits;
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
            string        strValue       = "";                   //The current selector
            List<string> list            = new List<string>();   //Combox will be filled by the list
            bool         bIsReadable     = false;                //Whether is implementary or not.
            
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
                    
                    //Get the current value of the function.
                    strValue = objIGXFeatureControl.GetEnumFeature(strFeatureName).GetValue();
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
                if (strTemp == strValue)
                {
                    cbEnum.SelectedIndex = i;
                    break;
                }
            }
        }


        /// <summary>
        /// Update the enable status of UI Widget
        /// </summary>
        private void __UpdateUI()
        {
            m_btn_OpenDevice.Enabled = m_bEnumDevices;
            m_btn_CloseDevice.Enabled = m_bIsOpen;
            m_btn_StartDevice.Enabled = m_bIsOpen && !m_bIsSnap;
            m_btn_StopDevice.Enabled = m_bIsSnap;

            m_cb_AutoWhite.Enabled = (m_bIsOpen && !m_bEnableAutoWhite);
            m_cb_AwbLight.Enabled = (m_bIsOpen && !m_bEnableAwbLight);
            m_cb_ChooseDevice.Enabled = m_bEnumDevices;

            m_ScrollBar_Sharpen.Enabled = (m_bIsOpen && !m_bEnableSharpness && m_checkBox_OpenSharpen.Checked);
            m_ScrollBar_Gamma.Enabled = (m_bIsOpen && !m_bEnableGamma);
            m_ScrollBar_Contrast.Enabled = m_bIsOpen;
            m_ScrollBar_Lightness.Enabled = m_bIsOpen;
            m_ScrollBar_Saturation.Enabled = (m_bIsOpen && !m_bEnableSaturation);

            // Update the enable status of edit widget
            m_txt_Sharpen.Enabled = (m_bIsOpen && !m_bEnableSharpness && m_checkBox_OpenSharpen.Checked);
            m_txt_Gamma.Enabled = (m_bIsOpen && !m_bEnableGamma);
            m_txt_Contrast.Enabled = m_bIsOpen;
            m_txt_Lightness.Enabled = m_bIsOpen;
            m_txt_Saturation.Enabled = (m_bIsOpen && !m_bEnableSaturation);

            // Update the enable status of check box widget
            m_CheckBox__BadPixelCorrect.Enabled = m_bIsOpen;
            m_checkBox_OpenColorCorrect.Enabled = (m_bIsOpen && !m_bEnableColorCorrect);
            m_checkBox_OpenDenoise.Enabled = (m_bIsOpen && !m_bEnableDenoise);
            m_checkBox_OpenSharpen.Enabled = (m_bIsOpen && !m_bEnableSharpness);
            m_checkBox_Accelerate.Enabled = m_bIsOpen;

        }

        /// <summary>
        /// Initialize camera
        /// </summary>
        private void __InitDevice()
        {

            if (null != m_objIGXFeatureControl)
            {
                //Setting the acquisition mode is continuous.
                m_objIGXFeatureControl.GetEnumFeature("AcquisitionMode").SetValue("Continuous");

                //Setting the trigger mode is off.
                m_objIGXFeatureControl.GetEnumFeature("TriggerMode").SetValue("Off");

                //Check whether the device supports the function of pixel color filter.
                m_bColorFilter = m_objIGXFeatureControl.IsImplemented("PixelColorFilter");

                // If supports the function of pixel color filter, get the value of pixel color filter.
                if (m_bColorFilter)
                {
                   m_strPixelColorFilter = m_objIGXFeatureControl.GetEnumFeature("PixelColorFilter").GetValue();
                }

                // Check whether the device supports the function of selecting AWBLampHouse.
                m_bAwbLampHouse = m_objIGXFeatureControl.IsImplemented("AWBLampHouse");

                __IsEnableDeviceColorCorrect();
                __IsEnableDeviceGamma();
                __IsEnableDeviceSharpness();
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
        /// Stop acquisition, close device and close stream
        /// </summary>
        private void __CloseAll()
        {
            try
            {
                //If the device doesn't stop to capture, it will be stopped to capture at first.
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
                //Stop the acquisition of stream, unregister the capture callback function and close the stream.
                if (null != m_objIGXStream)
                {
                    m_objIGXStream.StopGrab();
                    
                    //Unregister the capture callback function.
                    m_objIGXStream.UnregisterCaptureCallback();
                    m_objIGXStream.Close();
                    m_objIGXStream = null;
                    m_objIGXStreamFeatureControl = null;
                }

            }
            catch (Exception)
            {

            }

            //Close device
            __CloseDevice();
            m_bIsOpen = false;
        }


        /// <summary>
        /// Load window widget, initializing UI and the object of class IGXFactory.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void GxImageProcess_Load(object sender, EventArgs e)
        {
            try
            {
                //Update UI
                __UpdateUI();
                m_objIGXFactory = IGXFactory.GetInstance();
                m_objIGXFactory.Init();
                __InitDeviceCombox();

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// Initializes the select device drop-down box
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void __InitDeviceCombox()
        {
            try
            {
                m_listGXDeviceInfo = new List<IGXDeviceInfo>();
                m_objIGXFactory.UpdateDeviceList(200, m_listGXDeviceInfo);

                // Determine the number of devices currently connected
                if (m_listGXDeviceInfo.Count <= 0)
                {
                    MessageBox.Show("Not enumerated to device, please insert camera and restart the program!");
                    m_bEnumDevices = false;
                    __UpdateUI();
                    return;
                }
                else
                {
                    m_bEnumDevices = true;
                    __UpdateUI();

                    foreach(IGXDeviceInfo objDeviceInfo in m_listGXDeviceInfo)
                    {
                        m_cb_ChooseDevice.Items.Add(objDeviceInfo.GetDisplayName());
                    }

                    // The default drop-down list selects the first camera
                    m_cb_ChooseDevice.SelectedIndex = 0;
                }

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// The response function for the button of opening device.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_btn_OpenDevice_Click(object sender, EventArgs e)
        {
            try
            {
                //Close stream
                __CloseStream();

                // If the device has been opened then close it to ensure the camera can be opened again.
                __CloseDevice();

                int nSel = m_cb_ChooseDevice.SelectedIndex;
                if (nSel < 0)
                {
                    MessageBox.Show("Please insert camera!");
                    return;
                }

                //Open device
                m_objIGXDevice = m_objIGXFactory.OpenDeviceBySN(m_listGXDeviceInfo[nSel].GetSN(), GX_ACCESS_MODE.GX_ACCESS_EXCLUSIVE);
                m_objIGXFeatureControl = m_objIGXDevice.GetRemoteFeatureControl();

                __InitDevice();

                //Open stream
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

                m_objCfg = m_objIGXDevice.CreateImageProcessConfig();


                // Initialize UI widget
                __InitUI();

                m_objGxBitmap = new GxBitmap(m_objIGXDevice, m_pic_ShowImage);

                // Update the flag that indicates whether the device has been opened.
                m_bIsOpen = true;
                m_bEnumDevices = false;

                //Update the enable status of UI Widget
                __UpdateUI();

                // UI widgets in addition to the button of opening device are disabled before opening a device.
                // That will lead to  lack focus when press the tab. Setting the focus on the button of closing device after a device has been opened.
                m_btn_CloseDevice.Focus();

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            } 
        }

        /// <summary>
        /// The response function for the button of starting acquisition.
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
                //Open the stream
                if (null != m_objIGXStream)
                {
                    //The first parameter of the function RegisterCaptureCallback is belong to custom defined parameter(it's type must be quotative type).
                    //If want to use the parameter, the user can used it in the delegate function.
                    m_objIGXStream.RegisterCaptureCallback(this, __OnFrameCallbackFun);
                    m_objIGXStream.StartGrab();
                }

                //Send the command of starting acquisition.
                if (null != m_objIGXFeatureControl)
                {
                    m_objIGXFeatureControl.GetCommandFeature("AcquisitionStart").Execute();
                }
                m_bIsSnap = true;
                m_bEnumDevices = false;

                // Update the enable status of UI Widget
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// The response function for stopping acquisition
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
        /// The response function for the button of close device.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_btn_CloseDevice_Click(object sender, EventArgs e)
        {
            try
            {
                // Stop acquisition, close device and close stream.
                __CloseAll();
                
                m_lbl_IsAccelerate.Text = "";

                m_bEnumDevices = true;
                //Update UI
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// Delete resource after closing the window widget.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void GxImageProcess_FormClosed(object sender, FormClosedEventArgs e)
        {
            try
            {
                // Stop acquisition, close device and close stream.
                __CloseAll();

                //Execute the uninitialized function of GXFactory object
                if (null != m_objIGXFactory)
                {
                    m_objIGXFactory.Uninit();
                }
                
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        ///  Process image and display image.
        /// </summary>
        /// <param name="objIBaseData">The object of the IBaseData interface</param>
        public void ShowImageProcess(IBaseData objIBaseData)
        {
            GX_VALID_BIT_LIST emValidBits = GX_VALID_BIT_LIST.GX_BIT_0_7;
            try
            {
                if (null != m_objCfg)
                {
                    emValidBits = __GetBestValudBit(objIBaseData.GetPixelFormat());
                    m_objCfg.SetValidBit(emValidBits);

                    //In the function, at first, it will configure the image processing param
                    //to avoid updating UI death caused by image processing.
                    objIBaseData.ImageProcess(m_objCfg);
                }
            }
            catch (Exception)
            {
              
            }

            if (null != m_objCfg)
            {
                m_objGxBitmap.ShowImageProcess(m_objCfg, objIBaseData);
            }
        }

        /// <summary>
        /// The capturing event delegate function
        /// </summary>
        /// <param name="objUserParam">the custom defined parameter</param>
        /// <param name="objIFrameData">the object of the struct of image information</param>
        private void __OnFrameCallbackFun(object objUserParam, IFrameData objIFrameData)
        {
            try
            {
                GxImageProcess objGxImageProcess = objUserParam as GxImageProcess;
                objGxImageProcess.ShowImageProcess(objIFrameData);
            }
            catch (Exception)
            {

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

                // Update the enable status of UI Widget
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
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
        /// The response function for the combox that used to switch the auto white balance light source.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_cb_AwbLight_SelectedIndexChanged(object sender, EventArgs e)
        {
            try
            {
                string strValue = m_cb_AwbLight.Text;
                __SetEnumValue("AWBLampHouse", strValue, m_objIGXFeatureControl);

                // Update UI
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }


        /// <summary>
        /// The response function for the check box widget of defect correction
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_CheckBox__BadPixelCorrect_CheckedChanged(object sender, EventArgs e)
        {
            try
            {  
                m_objCfg.EnableDefectivePixelCorrect(m_CheckBox__BadPixelCorrect.Checked);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// The response function for the check box widget of image denoise.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_checkBox_OpenDenoise_CheckedChanged(object sender, EventArgs e)
        {
            try
            {
                //Only the color camera can supports image denoise function.
                if (m_bColorFilter)
                {
                    m_objCfg.EnableDenoise(m_checkBox_OpenDenoise.Checked);
                }   
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// The response function for the check box widget of color correction.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_checkBox_OpenColorCorrect_CheckedChanged(object sender, EventArgs e)
        {
            try
            {
                //Only the color camera can supports color correction function.
                if (m_bColorFilter)
                {
                    m_objCfg.EnableColorCorrection(m_checkBox_OpenColorCorrect.Checked);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// The response function for the check box widget of sharpen.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_checkBox_OpenSharpen_CheckedChanged(object sender, EventArgs e)
        {
            try
            {
                m_objCfg.EnableSharpen(m_checkBox_OpenSharpen.Checked);

                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }


        /// <summary>
        /// Set and display the value of sharpen.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_ScrollBar_Sharpen_ValueChanged(object sender, EventArgs e)
        {

            try
            {
                double dSharpen = 0.0;

                dSharpen = (double)m_ScrollBar_Sharpen.Value;
                dSharpen = dSharpen / 10.0;

                m_objCfg.SetSharpenParam(dSharpen);
                m_txt_Sharpen.Text = m_objCfg.GetSharpenParam().ToString();

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// Set and display the value of Gamma.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_ScrollBar_Gamma_ValueChanged(object sender, EventArgs e)
        {
            try
            {
                double dGamma = 0.0;
                dGamma = (double)m_ScrollBar_Gamma.Value;
                dGamma = dGamma / 10.0;

                m_objCfg.SetGammaParam(dGamma);
                m_txt_Gamma.Text = m_objCfg.GetGammaParam().ToString();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// Set and display the value of contrast.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_ScrollBar_Contrast_ValueChanged(object sender, EventArgs e)
        {
            try
            {
                m_objCfg.SetContrastParam(m_ScrollBar_Contrast.Value);
                m_txt_Contrast.Text = m_objCfg.GetContrastParam().ToString();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// Set and display the value of lightness.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_ScrollBar_Lightness_ValueChanged(object sender, EventArgs e)
        {
            try
            {
                m_objCfg.SetLightnessParam(m_ScrollBar_Lightness.Value);
                m_txt_Lightness.Text = m_objCfg.GetLightnessParam().ToString();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// Set and display the value of saturation.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_ScrollBar_Saturation_ValueChanged(object sender, EventArgs e)
        {
            try
            {
                //Only the color camera can supports setting saturation.
                if (m_bColorFilter)
                {
                    m_objCfg.SetSaturationParam(m_ScrollBar_Saturation.Value);
                    m_txt_Saturation.Text = m_objCfg.GetSaturationParam().ToString();
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
                string strValue = ""; //he value of auto white balance
                Int32 i = 0;          //Loop variables

                // If the value of auto white balance is once will be set , the actual value of auto white balance will become off automatically after setting once successfuly.
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

        /// <summary>
        /// The response function for the check box widget of whether is enabled algorithm acceleration.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_checkBox_Accelerate_CheckedChanged(object sender, EventArgs e)
        {
            try
            {
                int nHeight = (int)m_objIGXFeatureControl.GetIntFeature("Height").GetValue();

                //Check the image height whether is a integer multiple of 4 and the enable status of algorithm acceleration.
                bool bFlag = ((nHeight % 4) != 0) && m_checkBox_Accelerate.Checked; 
                if (bFlag)
                {
                    MessageBox.Show("If want use CPU algorithm acceleration, the image height must be a integer multiple of 4!");
                    m_checkBox_Accelerate.Checked = false;
                    return;
                }
                m_objCfg.EnableAccelerate(m_checkBox_Accelerate.Checked);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// Gets whether the device opens color correction
        /// </summary>
        private void __IsEnableDeviceColorCorrect()
        {
            m_bEnableColorCorrect = false;
            bool bIsImplement = m_objIGXFeatureControl.IsImplemented("ColorTransformationEnable");
            if(false == bIsImplement)
            {
                return;
            }
            IBoolFeature pBoolFeature = m_objIGXFeatureControl.GetBoolFeature("ColorTransformationEnable");
            m_bEnableColorCorrect = pBoolFeature.GetValue();
        }

        /// <summary>
        /// Gets whether the device opens the Gamma enabled
        /// </summary>
        private void __IsEnableDeviceGamma()
        {
            m_bEnableGamma = false;
            bool bIsImplement = m_objIGXFeatureControl.IsImplemented("GammaEnable");
            if(false == bIsImplement)
            {
                return;
            }

            IBoolFeature pBoolFeature = m_objIGXFeatureControl.GetBoolFeature("GammaEnable");

            m_bEnableGamma = pBoolFeature.GetValue();
        }

        /// <summary>
        /// Gets whether the device opens sharpening enable
        /// </summary>
        private void __IsEnableDeviceSharpness()
        {
            m_bEnableSharpness = false;
            bool bIsImplement = m_objIGXFeatureControl.IsImplemented("SharpnessMode");
            if (false == bIsImplement)
            {
                return;
            }

            IEnumFeature pEnumFeature = m_objIGXFeatureControl.GetEnumFeature("SharpnessMode");
            string strValue = pEnumFeature.GetValue();

            if (true == strValue.Equals("Off"))
            {
                m_bEnableSharpness = false;
            }
            else if (true == strValue.Equals("On"))
            {
                m_bEnableSharpness = true;
            }
        }
    }
}
