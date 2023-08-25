using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using GxIAPINET;

namespace GxGigeRecovery
{
   public  class CGigeRecovery
    {

        GX_DEVICE_OFFLINE_CALLBACK_HANDLE m_hCB = null;                           ///< The handle for OffLine callback event
        string m_strPath = "";                                                    ///< The current path for file
        string m_strSavePath = "";                                                ///< The path for saving device configuration parameters
        string m_strMACAddress = "";                                              ///< The MAC address for device
        bool m_bIsOffLine = false;                                                ///< The flag indicates whether the device is offline
        bool m_bIsOpen = false;                                                   ///< The flag indicates whether the first device has been opened
        bool m_bIsSanp = false;                                                   ///< The flag indicates whether AcquisitionStart command has been send
        IGXFactory m_objIGXFactory = null;                                        ///< The handle for factory
        IGXDevice m_objIGXDevice = null;                                          ///< The handle for device
        IGXStream m_objIGXStream = null;                                          ///< The handle for stream
        IGXFeatureControl m_objIGXFeatureControl = null;                          ///< The handle for feature control
        IGXFeatureControl m_objIGXStreamFeatureControl = null;                    ///< The object for stream feature control



        /// <summary>
        /// Construction
        /// </summary>
        public CGigeRecovery()
        {
            try
            {
                // Before using any GxIAPINET methods, the GxIAPINET must be initialized.
                m_objIGXFactory = IGXFactory.GetInstance();
                m_objIGXFactory.Init();
                m_strPath = System.Environment.CurrentDirectory + "\\";
                m_strSavePath = m_strPath + "ConfigFile.ini";
            }
            catch (CGalaxyException ex)
            {
                string strErrorInfo = "error code:" + ex.GetErrorCode().ToString() + "error information:" + ex.Message;
                Console.WriteLine(strErrorInfo);
                System.Environment.Exit(System.Environment.ExitCode);
            }
        }

        /// <summary>
        /// The external interface function for OffLine recovery
        /// </summary>
        public void GigeRecovery()
        {
            bool bReturn = false;
            try
            {
                // Open device
                bReturn = __OnOpenDevice();
                if (!bReturn)
                {
                    return;
                }

                // Start acquisition
                bReturn = __OnStartSnap();
                if (!bReturn)
                {
                    __OnCloseDevice();
                    return;
                }

                // Continuous acquisition 
                __Acquisition();

                // Stop acquisition
                __OnStopSnap();

                // Close device
                __OnCloseDevice();
            }
            catch (CGalaxyException ex)
            {
                string strErrorInfo = "error code:" + ex.GetErrorCode().ToString() + "error information:" + ex.Message;
                Console.WriteLine(strErrorInfo);
            }
        }

        /// <summary>
        /// Open device
        /// </summary>
        /// <returns></returns>
        private bool __OnOpenDevice()
        {
            bool bReturn    = false;    
            bool bOpenState = false;   
            try
            {
                // Enumerate all camera devices
                Console.Write("====================CGigeRecovery::__OnOpenDevice()====================\n");
                List<IGXDeviceInfo> listGXDeviceInfo = new List<IGXDeviceInfo>();
                m_objIGXFactory.UpdateDeviceList(1000, listGXDeviceInfo);

                // Check if found any device
                if (listGXDeviceInfo.Count <= 0)
                {
                    Console.Write("<NO device>\n");
                    return false;
                }

                IGXDeviceInfo objIGXDeviceInfo = listGXDeviceInfo[0];

                // Check whether the device is Gige
                if (objIGXDeviceInfo.GetDeviceClass() != GX_DEVICE_CLASS_LIST.GX_DEVICE_CLASS_GEV)
                {
                    Console.Write("<The device is not GEV>");
                    return false;
                }

                // Get MAC address
                m_strMACAddress = objIGXDeviceInfo.GetMAC();

                Console.Write("<Open device by MAC:{0}\n", m_strMACAddress);
                m_objIGXDevice = m_objIGXFactory.OpenDeviceByMAC (m_strMACAddress, GX_ACCESS_MODE.GX_ACCESS_EXCLUSIVE);
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

                // Initialize the device parameters
                Console.Write("<Initialize the device parameters>\n");
                bReturn = __InitParam();
                if (!bReturn)
                {
                    return false;
                }

                // Export the parameter config file 
                Console.Write("<Export config file>\n");
                if (null != m_objIGXDevice)
                {
                    m_objIGXDevice.ExportConfigFile(m_strSavePath);
                }

                // Register the OffLine callback function
                Console.Write("<Register device Offline callback>\n");
                if (null != m_objIGXDevice)
                {
                    m_hCB = m_objIGXDevice.RegisterDeviceOfflineCallback(null, __OnDeviceOfflineCallbackFun);
                }

                bOpenState = true;

                m_bIsOpen = true;


            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                bOpenState = false;
            }
            return bOpenState;
        }

        
        /// <summary>
        /// Initialize the device parameters
        /// </summary>
        /// <returns>true: success, false: fail</returns>
        private bool __InitParam()
        {
            bool bInitParamState = false;
            try
            {
                if (null != m_objIGXFeatureControl)
                {
                    // Set the continuous frame acquisition mode
                    m_objIGXFeatureControl.GetEnumFeature("AcquisitionMode").SetValue("Continuous");

                    // Close the TriggerMode
                    m_objIGXFeatureControl.GetEnumFeature("TriggerMode").SetValue("Off");

                    //Set heartbeat timeout for 1s
                    //If you build this sample in debug mode and run it using a GigE camera device,the device will be set heartbeat timeout to 5 minutes
                    //This is done to prevent the camera's heartbeat timeout from affecting the program's debugging and single stepping of execution, This also means that the camera can not be disconnected for 5 minutes unless the camera is powered off and on 
                    //To work around this, the heartbeat timeout will be set to 1000 ms 
                    m_objIGXFeatureControl.GetIntFeature("GevHeartbeatTimeout").SetValue(1000);

                    if (null != m_objIGXStreamFeatureControl)
                    {
                        //Set StreamBufferHandlingMode
                        m_objIGXStreamFeatureControl.GetEnumFeature("StreamBufferHandlingMode").SetValue("OldestFirst");

                        bInitParamState = true;
                    }
                    else
                    {
                        bInitParamState = false;
                    }
                }
               
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                bInitParamState = false;
            }
            return bInitParamState;
        }

        /// <summary>
        /// Start acquisition
        /// </summary>
        /// <returns>true: success, false: fail</returns>
        private bool __OnStartSnap()
        {
            bool bStartSnapState = false;
            try
            {
                Console.WriteLine("====================CGigeRecovery::__OnStartSnap()====================");

                Console.WriteLine("<Send start snap command to device>");

                // Start stream channel acquisition
                if (null != m_objIGXStream)
                {
                    m_objIGXStream.StartGrab();
                }

                // Send acquisitionStart command
                if (null != m_objIGXFeatureControl)
                {
                    m_objIGXFeatureControl.GetCommandFeature("AcquisitionStart").Execute();
                }

                m_bIsSanp = true;
                bStartSnapState = true;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                bStartSnapState = false;
            }
            return bStartSnapState;
        }

        /// <summary>
        /// Stop acquisition
        /// </summary>
        /// <returns>true: success, false: fail</returns>
        private bool __OnStopSnap()
        {
            bool bStopSnapState = false;
            try
            {
                Console.WriteLine("====================CGigeRecovery::__OnStopSnap()====================");
                Console.WriteLine("<Send stop snap command to device>");

                // Send acquisitionStop command
                if (null != m_objIGXFeatureControl)
                {
                    m_objIGXFeatureControl.GetCommandFeature("AcquisitionStop").Execute();
                }

                // Stop stream channel acquisition
                if (null != m_objIGXStream)
                {
                    m_objIGXStream.StopGrab();
                }

                m_bIsSanp = false;
                bStopSnapState = true;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                bStopSnapState = false;
            }
            return bStopSnapState;
        }

        /// <summary>
        /// Close device
        /// </summary>
        /// <returns></returns>
        private bool __OnCloseDevice()
        {
            try
            {
                Console.WriteLine("====================CGigeRecovery::__OnCloseDevice()====================");
               
                 // Close stream
                if (null != m_objIGXStream)
                {
                    m_objIGXStream.Close();
                    m_objIGXStream = null;
                    m_objIGXStreamFeatureControl = null;
                }

                // // Unregister the device OffLine callback function 
                Console.WriteLine("<Unregister device Offline callback>");
                if (null != m_objIGXDevice)
                {
                    m_objIGXDevice.UnregisterDeviceOfflineCallback(m_hCB);
                }

                // Close device
                Console.WriteLine("<Close device>");
                if (null != m_objIGXDevice)
                {
                    m_objIGXDevice.Close();
                    m_objIGXDevice = null;
                }

                m_bIsOpen = false;

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            } 
            return false;
        }

        /// <summary>
        /// Continuous acquisition
        /// </summary>
        private void __Acquisition()
        {
            IImageData objIImageData = null; //GetImage acquisition object
            Console.WriteLine("====================CGigeRecovery::__Acquisition()====================");
            Console.WriteLine("<Press any key to stop acquisition>");

            // Check whether the keyboard is pressed
            while (!System.Console.KeyAvailable)
            {
                try
                {
                    if (m_bIsOffLine)         // The device is OffLine
                    {
                        // Deal with offline
                        __ProcessOffline();

                        // Device recovery
                        __Recovery();
                    }
                    else                     // The device is OnLine
                    {
                        if (null != m_objIGXStream)
                        {
                            objIImageData = m_objIGXStream.GetImage(500);
                            Console.WriteLine("Successfully get Image");
                            
                            // Release resource
                            if (null != objIImageData)
                            {
                                objIImageData.Destroy();
                                objIImageData = null; 
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                } 
            }

            Console.ReadKey();
        }

        /// <summary>
        /// Deal with the OffLine event
        /// </summary>
        private void __ProcessOffline()
        {

            Console.Write("**********************Process Offline**********************\r");
            try
            {
                // If the device is acquiring image then stop it
                if (m_bIsSanp)
                {
                    Console.Write("\n<Send stop snap command to device>\n");

                    // Send acquisitionstop command
                    if (null != m_objIGXFeatureControl)
                    {
                        m_objIGXFeatureControl.GetCommandFeature("AcquisitionStop").Execute();
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            finally
            {
                m_objIGXFeatureControl = null;
            }

            try
            {

                // If the device is acquiring image then stop it
                if (m_bIsSanp)
                {

                    // Stop stream channel acquisition
                    if (null != m_objIGXStream)
                    {
                        m_objIGXStream.StopGrab();
                        m_objIGXStream.Close();
                        m_objIGXStream = null;
                        m_objIGXStreamFeatureControl = null;
                    }
                }

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            finally
            {
                m_bIsSanp = false;
            }

            try
            {
                // If the device is opened then close it.
                if (m_bIsOpen)
                {
                    // Unregister the OffLine callback function
                    Console.WriteLine("<Unregister device Offline callback>");
                    if (null != m_objIGXDevice)
                    {
                        m_objIGXDevice.UnregisterDeviceOfflineCallback(m_hCB);
                    }

                    // Close device
                    Console.WriteLine("<Close device>");
                    if (null != m_objIGXDevice)
                    {
                        m_objIGXDevice.Close();
                        m_objIGXDevice = null;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            finally
            {
                m_bIsOpen = false;
            }
        }

        /// <summary>
        /// Recovery the device from OffLine status
        /// </summary>
        private void __Recovery()
        {
            try
            {
                // Enumerate all camera devices
                Console.Write("**********************Recovery**********************\r");
                List<IGXDeviceInfo> listGXDeviceInfo = new List<IGXDeviceInfo>();
                m_objIGXFactory.UpdateDeviceList(1000, listGXDeviceInfo);

                // Check if found any device
                if (listGXDeviceInfo.Count <= 0)
                {
                    return ;
                }

                // Check whether the device has been opened
                if (null != m_objIGXDevice)
                {
                    m_objIGXDevice.Close();
                    m_objIGXDevice = null;
                }

                // Open device by MAC address
                Console.WriteLine("<Open device by MAC:{0}", m_strMACAddress);
                m_objIGXDevice = m_objIGXFactory.OpenDeviceByMAC(m_strMACAddress, GX_ACCESS_MODE.GX_ACCESS_EXCLUSIVE);
                m_objIGXFeatureControl = m_objIGXDevice.GetRemoteFeatureControl();


                // Open stream
                if (null != m_objIGXDevice)
                {
                    m_objIGXStream = m_objIGXDevice.OpenStream(0);
                    m_objIGXStreamFeatureControl = m_objIGXStream.GetFeatureControl();
                }

                // Import config file
                Console.WriteLine("<Import config file>");
                if (null != m_objIGXDevice)
                {
                    m_objIGXDevice.ImportConfigFile(m_strSavePath);
                }

                // Register the OffLine callback function
                Console.WriteLine("<Register device Offline callback>");
                if (null != m_objIGXDevice)
                {
                    m_hCB = m_objIGXDevice.RegisterDeviceOfflineCallback(null, __OnDeviceOfflineCallbackFun);
                }
     
                m_bIsOpen = true;

                Console.WriteLine("<Send start snap command to device>");

                // Start stream channel acquisition
                if (null != m_objIGXStream)
                {
                    m_objIGXStream.StartGrab();
                }

                // Send acquisitionStart command
                if (null != m_objIGXFeatureControl)
                {
                    m_objIGXFeatureControl.GetCommandFeature("AcquisitionStart").Execute();
                }

                m_bIsSanp = true;
                m_bIsOffLine = false;

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return;
            }
        }

        /// <summary>
        /// This is a offline callback function that demonstrates how to record the device is offline
        /// </summary>
        /// <param name="pUserParam">user private param</param>
        private void __OnDeviceOfflineCallbackFun(object pUserParam)
        {
            try
            {
                m_bIsOffLine = true;
                Console.Write("**********************Device offline**********************\n");
            }
            catch (Exception)
            {

            }
        }
    }
}
