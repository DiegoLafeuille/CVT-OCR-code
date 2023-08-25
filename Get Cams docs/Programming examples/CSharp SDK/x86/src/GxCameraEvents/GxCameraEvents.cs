using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using GxIAPINET;
using GxIAPINET.Sample.Common;

namespace GxCameraEvents
{
    public partial class GxCameraEvents : Form
    {
        bool m_bIsOpen = false;                                               ///< The flag indicates whether the first device has been opened 
        bool m_bIsSnap = false;                                               ///< The flag indicates whether AcquisitionStart command has been send 
        IGXFactory m_objIGXFactory = null;                                    ///< The handle for factory 
        IGXDevice m_objIGXDevice = null;                                      ///< The handle for device 
        IGXStream m_objIGXStream = null;                                      ///< The object for stream
        IGXFeatureControl m_objIGXFeatureControl = null;                      ///< The object for device feature control
        IGXFeatureControl m_objIGXStreamFeatureControl = null;                ///< The object for stream feature control
        GX_FEATURE_CALLBACK_HANDLE m_hFeatureCallback = null;                 ///< The pointer for ExposureEndEvent callback
        CStopWatch m_objImageTime = null;                                     ///< The stopwatch for receiving image
        CStopWatch m_objShutterTime = null;                                   ///< The stopwatch for completing the exposure event
        bool m_bIsTrigValid = true;                                           ///< The flag indicates Valid flag for trigger: sets this flag to false when a trigger is executing    
                               
        public GxCameraEvents()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Update UI
        /// </summary>
        private void __UpdateUI()
        {
            m_btn_OpenDevice.Enabled = !m_bIsOpen;
            m_btn_CloseDevice.Enabled = m_bIsOpen;
            m_btn_StartDevice.Enabled = m_bIsOpen && !m_bIsSnap;
            m_btn_StopDevice.Enabled = m_bIsSnap;
            m_btn_SoftTriggerCommand.Enabled = m_bIsOpen && m_bIsSnap;
        }

        /// <summary>
        /// initialize device param
        /// </summary>
        private void __InitDevice()
        {
            if (null != m_objIGXFeatureControl)
            {
                // Set the continuous frame acquisition mode
                m_objIGXFeatureControl.GetEnumFeature("AcquisitionMode").SetValue("Continuous");

                // Set the TriggerMode on
                m_objIGXFeatureControl.GetEnumFeature("TriggerMode").SetValue("On");

                // Set the TriggerSource to SoftTrigger
                m_objIGXFeatureControl.GetEnumFeature("TriggerSource").SetValue("Software");

                // Select the ExposureEndEvent
                m_objIGXFeatureControl.GetEnumFeature("EventSelector").SetValue("ExposureEnd");

                // Enable the EventExposureEnd
                m_objIGXFeatureControl.GetEnumFeature("EventNotification").SetValue("On");

            }
        }


        /// <summary>
        /// Initialize UI and GxIAPINET
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void GxCameraEvents_Load(object sender, EventArgs e)
        {
            try
            {
                // Update UI
                __UpdateUI();

                m_objImageTime = new CStopWatch();
                m_objShutterTime = new CStopWatch();
                m_objIGXFactory = IGXFactory.GetInstance();
                m_objIGXFactory.Init();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// Close stream
        /// </summary>
        private void __CloseStream()
        {
            try
            {
                // Close stream
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
        /// Close Device
        /// </summary>
        private void __CloseDevice()
        {
            try
            {
                // Close Device
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
        /// AcquisitionStop and Close stream and Close device
        /// </summary>
        private void __CloseAll()
        {
            try
            {
                // Check whether the device has been stoped acquisition
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

            m_bIsSnap = false;

            try
            {
                // Check whether the stream  has been closed
                if (null != m_objIGXStream)
                {
                    // Stop Stream acquisition
                    m_objIGXStream.StopGrab();

                    // Unregister the ExposureEndEvent callback function
                    m_objIGXStream.UnregisterCaptureCallback();

                    // Close steam
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
                // Check whether the the ExposureEndEvent  has been Unregistered
                if (null != m_objIGXFeatureControl)
                {
                    if (null != m_hFeatureCallback)
                    {
                        m_objIGXFeatureControl.UnregisterFeatureCallback(m_hFeatureCallback);
                        m_objIGXFeatureControl = null;
                    }
                }
            }
            catch (Exception)
            {
            }

            // Close device
            __CloseDevice();
            m_bIsOpen = false;
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

                // Check whether the device has been stoped acquisition
                __CloseStream();

                // Check whether the device has been closed
                __CloseDevice();

                m_objIGXFactory.UpdateDeviceList(200, listGXDeviceInfo);

                if (listGXDeviceInfo.Count <= 0)
                {
                    MessageBox.Show("No device found!");
                    return;
                }

                // Open the first device in the list
                m_objIGXDevice = m_objIGXFactory.OpenDeviceBySN(listGXDeviceInfo[0].GetSN(), GX_ACCESS_MODE.GX_ACCESS_EXCLUSIVE);
                m_objIGXFeatureControl = m_objIGXDevice.GetRemoteFeatureControl();


                __InitDevice();

                // Open stream
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

                m_bIsOpen = true;

                // Update UI
                __UpdateUI();

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }      
        }

        /// <summary>
        /// AcquisitionStart
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_btn_StartDevice_Click(object sender, EventArgs e)
        {
            try
            {
                // Register ExposureEndEvent callback function
                if (null != m_objIGXFeatureControl)
                {
                    m_hFeatureCallback = m_objIGXFeatureControl.RegisterFeatureCallback("EventExposureEnd", null, __OnFeatureCallbackFun);
                }

                if (null != m_objIGXStreamFeatureControl)
                {
                    //Set StreamBufferHandlingMode
                    m_objIGXStreamFeatureControl.GetEnumFeature("StreamBufferHandlingMode").SetValue("OldestFirst");
                }
                //Start stream channel acquisition
                if (null != m_objIGXStream)
                {
                    // Register the CaptureCallback function
                    m_objIGXStream.RegisterCaptureCallback(null, __OnFrameCallbackFun);
                    m_objIGXStream.StartGrab();
                }

                // Send AcquisitionStart command
                if (null != m_objIGXFeatureControl)
                {
                    m_objIGXFeatureControl.GetCommandFeature("AcquisitionStart").Execute();
                }
                m_bIsSnap = true;
                m_bIsTrigValid = true;

                // Update UI
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// Send TriggerSoftware command
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_btn_SoftTriggerCommand_Click(object sender, EventArgs e)
        {
            try
            {
                // If you click the trigger button again while the trigger callback is executing, the click will be blocked
                if (!m_bIsTrigValid)
                {
                    return;
                }
                else
                {
                    m_bIsTrigValid = false;
                }

                // Get the time of sending TriggerSoftware command 
                string strTimeNow = "\r\nThe time of sending TriggerSoftware command:" + DateTime.Now.Hour.ToString("D2")
                                    + "h:" + DateTime.Now.Minute.ToString("D2")
                                    + "m:" + DateTime.Now.Second.ToString("D2")
                                    + "s:" + DateTime.Now.Millisecond.ToString("D3") + "ms" + "\r\n";
                m_txt_TimeLog.AppendText(strTimeNow);

                // Send TriggerSoftware command
                if (null != m_objIGXFeatureControl)
                {
                    // Start timing for receiving image
                    m_objImageTime.Start();

                    // Start timing for recording the process of ExposureEndEvent
                    m_objShutterTime.Start();

                    m_objIGXFeatureControl.GetCommandFeature("TriggerSoftware").Execute();
                }

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
                m_bIsTrigValid = true;
            }  
        }

        /// <summary>
        /// AcquisitionStop
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_btn_StopDevice_Click(object sender, EventArgs e)
        {
            try
            {
                // Send AcquisitionStop command
                if (null != m_objIGXFeatureControl)
                {
                    m_objIGXFeatureControl.GetCommandFeature("AcquisitionStop").Execute();
                }

                // Stop stream channel acquisition
                if (null != m_objIGXStream)
                {
                    m_objIGXStream.StopGrab();

                    // Unregister the CaptureCallback function
                    m_objIGXStream.UnregisterCaptureCallback();
                }

                // Unregister the ExposureEndEvent callback function
                if (null != m_objIGXFeatureControl)
                {
                    m_objIGXFeatureControl.UnregisterFeatureCallback(m_hFeatureCallback);
                }

                m_bIsSnap = false;

                // Update UI
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }  
        }

        /// <summary>
        /// Close Devcie
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_btn_CloseDevice_Click(object sender, EventArgs e)
        {
            try
            {
                // Close All
                __CloseAll();

                // Update UI
                __UpdateUI();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// Close Window,Close stream,Close device,Release GxIAPINET
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void GxCameraEvents_FormClosed(object sender, FormClosedEventArgs e)
        {
            try
            {
                // Close All
                __CloseAll();

                // Release GxIAPINET
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
        /// This is a acquisition callback function that demonstrates how to record the interval that from sending a TriggerSoftware command to receiving a image.
        /// </summary>
        /// <param name="objUserParam">user private param</param>
        /// <param name="objIFrameData">image information</param>
        private void __OnFrameCallbackFun(object objUserParam, IFrameData objIFrameData)
        {
            if (InvokeRequired)
            {
                this.BeginInvoke(new CaptureDelegate(__OnFrameCallbackFun), new object[] { objUserParam, objIFrameData });
                return;
            }
            double dElapsedtime = 0;

            try
            {
                // Stop timing
                dElapsedtime = m_objImageTime.Stop();
                string strImageTime = "The interval that from sending a TriggerSoftware command to receiving an image:" + dElapsedtime.ToString("F2") + "ms" + "\r\n";
                __ShowTimeLog(strImageTime);
            }
            catch (Exception)
            {

            }

            m_bIsTrigValid = true;
        }


        /// <summary>
        /// This is a event callback function that demonstrates how to process the ExposureEndEvent and record the interval that from sending a TriggerSoftware command to receiving a ExposureEndEvent
        /// </summary>
        /// <param name="objUserParam">user private param</param>
        /// <param name="strFeatureName">Event name</param>
        private void __OnFeatureCallbackFun(object objUserParam, string strFeatureName)
        {
           
            double dElapsedtime = 0;
            try
            {
                if (InvokeRequired)
                {
                    this.BeginInvoke(new FeatureDelegate(__OnFeatureCallbackFun), new object[] { objUserParam, strFeatureName });
                    return;
                }

                if ("EventExposureEnd" == strFeatureName)
                {
                    // Stop timing
                    dElapsedtime = m_objShutterTime.Stop();
                    string strShutterTime = "The interval that from sending a TriggerSoftware command to receiving a ExposureEndEvent:" + dElapsedtime.ToString("F2") + "ms" + "\r\n";
                    __ShowTimeLog(strShutterTime);
                }
            }
            catch (Exception)
            {

            }
        }

        /// <summary>
        /// Update time to Edit control
        /// </summary>
        /// <param name="strTimeLogInfo">Display information</param>
        private void __ShowTimeLog(string strTimeLogInfo)
        {
            m_txt_TimeLog.AppendText(strTimeLogInfo); 
        }

    }
}
