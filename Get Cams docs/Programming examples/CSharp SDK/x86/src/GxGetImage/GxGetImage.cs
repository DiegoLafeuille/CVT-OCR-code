using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Diagnostics;
using GxIAPINET;
using GxIAPINET.Sample.Common;

namespace GxGetImage
{
    public partial class GxGetImage : Form
    {
        bool m_bIsOpen = false;                                 ///< The flag indicates whether the first device has been opened
        bool m_bIsSnap = false;                                 ///< The flag indicates whether AcquisitionStart command has been send 
        IGXFactory m_objIGXFactory = null;                      ///< The handle for factory
        IGXDevice m_objIGXDevice = null;                        ///< The handle for device
        IGXStream m_objIGXStream = null;                        ///< The handle for stream
        IGXFeatureControl m_objIGXFeatureControl = null;        ///< The handle for feature control
        IGXFeatureControl m_objIGXStreamFeatureControl = null;  ///< The object for stream feature control
        CStatistics m_objStatistic = new CStatistics();         ///< Statistics
        CStopWatch m_objStopTime = new CStopWatch();            ///< Stopwatch
        GxBitmap   m_objGxBitmap = null;                        ///< The handle for bitmap

        public GxGetImage()
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
            m_txt_TimeOut.Enabled = m_bIsOpen && m_bIsSnap;
        }

        /// <summary>
        /// Initialize device param
        /// </summary>
        private void __InitDevice()
        {
            if (null != m_objIGXFeatureControl)
            {
                //Set the continuous frame acquisition mode
                m_objIGXFeatureControl.GetEnumFeature("AcquisitionMode").SetValue("Continuous");

                //Set the TriggerMode on
                m_objIGXFeatureControl.GetEnumFeature("TriggerMode").SetValue("On");

                //Set the TriggerSource to SoftTrigger
                m_objIGXFeatureControl.GetEnumFeature("TriggerSource").SetValue("Software");
            }
        }

        /// <summary>
        ///  Update statistical time data in UI
        /// </summary>
        /// <param name="dData">Time data</param>
        private void __UpdateStatisticalData(double dData)
        {
            m_objStatistic.AddStatisticalData(dData);

            // Show the average value
            m_txt_AveTime.Text = m_objStatistic.GetAverage().ToString("F3");

            // Show the max value 
            m_txt_MaxTime.Text = m_objStatistic.GetMax().ToString("F3");

            // Show the min value
            m_txt_MinTime.Text = m_objStatistic.GetMin().ToString("F3");
        }

        /// <summary>
        /// Initialize UI and GxIAPINET
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void GxGetImage_Load(object sender, EventArgs e)
        {
            try
            {
                // Update UI
                __UpdateUI();

                // Before using any GxIAPINET methods, the GxIAPINET must be initialized.
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
        /// Close all
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
                // Stop stream channel acquisition and close stream
                if (null != m_objIGXStream)
                {
                    m_objIGXStream.StopGrab();
                    m_objIGXStream.Close();
                    m_objIGXStream = null;
                    m_objIGXStreamFeatureControl = null;
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

                // Close stream
                __CloseStream();

                // If the device is opened then close it to ensure the camera could open again.
                __CloseDevice();

                // Enumerate all camera devices
                m_objIGXFactory.UpdateDeviceList(200, listGXDeviceInfo);

                // Check if found any device
                if (listGXDeviceInfo.Count <= 0)
                {
                    MessageBox.Show("No devices found!");
                    return;
                }

                //Open the first found device 
                m_objIGXDevice = m_objIGXFactory.OpenDeviceBySN(listGXDeviceInfo[0].GetSN(), GX_ACCESS_MODE.GX_ACCESS_EXCLUSIVE);
                m_objIGXFeatureControl = m_objIGXDevice.GetRemoteFeatureControl();
               

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

                __InitDevice();

                m_objGxBitmap = new GxBitmap(m_objIGXDevice, m_pic_ShowImage);

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
        /// Close device and stream
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_btn_CloseDevice_Click(object sender, EventArgs e)
        {
            try
            {
                // Reset statistical time count
                m_objStatistic.Reset();

                // close stream and device
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
        /// Start acquisition
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

                // Start stream channel acquisition
                if (null != m_objIGXStream)
                {
                    m_objIGXStream.StartGrab();
                }

                // Send AcquisitionStart command
                if (null != m_objIGXFeatureControl)
                {
                    m_objIGXFeatureControl.GetCommandFeature("AcquisitionStart").Execute();
                }
                m_bIsSnap = true;

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
                // Send AcquisitionStop command
                if (null != m_objIGXFeatureControl)
                {
                    m_objIGXFeatureControl.GetCommandFeature("AcquisitionStop").Execute();
                }

                // Stop stream channel acquisition
                if (null != m_objIGXStream)
                {
                    m_objIGXStream.StopGrab();
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
        /// TriggerSoftware command,acquire an image
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_btn_SoftTriggerCommand_Click(object sender, EventArgs e)
        {
            try
            {
                IImageData objIImageData = null;
                double dElapsedtime = 0;
                uint nTimeout = 500;

                if (!string.IsNullOrEmpty(m_txt_TimeOut.Text))
                {
                    try
                    {
                        nTimeout = Convert.ToUInt32(m_txt_TimeOut.Text);
                    }
                    catch (Exception)
                    {
                        m_txt_TimeOut.Text = (500).ToString();
                        MessageBox.Show("Please enter the correct number of valid!");
                        return;
                    }
                }

                m_txt_TimeOut.Text = nTimeout.ToString();

                //Flush image queues to clear out-of-date images
                if (null != m_objIGXStream)
                {
                    m_objIGXStream.FlushQueue();
                }

                //Send TriggerSoftware commands
                if (null != m_objIGXFeatureControl)
                {
                    m_objIGXFeatureControl.GetCommandFeature("TriggerSoftware").Execute();
                }

                //Get image
                if (null != m_objIGXStream)
                {
                    //Start stopwatch
                    m_objStopTime.Start();

                    objIImageData = m_objIGXStream.GetImage(nTimeout);

                    //Stop stopwatch and get the ElapsedTime
                    dElapsedtime = m_objStopTime.Stop();
                }

                m_objGxBitmap.Show(objIImageData);

                if (null != objIImageData)
                {
                    // Release resource
                    objIImageData.Destroy();
                }
                
                //Update statistics time count
                __UpdateStatisticalData(dElapsedtime);
               
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        /// <summary>
        /// Timeout can only enter numbers
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_txt_TimeOut_KeyPress(object sender, KeyPressEventArgs e)
        {
            if (!char.IsNumber(e.KeyChar) && e.KeyChar != 8)
            {
                e.Handled = true;
            }
        }

        /// <summary>
        /// Close all
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void GxGetImage_FormClosed(object sender, FormClosedEventArgs e)
        {
            try
            {
                // Close stream and stream
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
    }
}
