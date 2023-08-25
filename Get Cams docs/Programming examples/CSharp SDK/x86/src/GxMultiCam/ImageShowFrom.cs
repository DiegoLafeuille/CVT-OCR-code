using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
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
    public partial class ImageShowFrom : Form
    {

        CFps m_objCFps = new CFps();                                  ///<The object for statistical FPS
        IGXDevice m_objIGXDevice = null;                              ///<The handle for device 
        IGXStream m_objIGXStream = null;                              ///<The object for stream
        IGXFeatureControl m_objIGXFeatureControl = null;              ///<The object for device feature control 
        IGXFeatureControl m_objIGXStreamFeatureControl = null;        ///<The object for stream feature control
        string m_strSN = "";                                          ///<Serial number
        bool m_bIsColor = false;                                      ///<The flag indicates whether the device is supported color filter or not.
        byte[]            m_byMonoBuffer               = null;        ///<The buffer for black and white camera
        byte[] m_byColorBuffer = null;                                ///<The buffer for color camera       
        int               m_nWidth                     = 0;           ///<Image width
        int               m_nHeigh                     = 0;           ///<Image height
        Bitmap m_bitmap = null;                                       ///<The object for stream bitmap
        const uint PIXEL_FORMATE_BIT = 0x00FF0000;                    ///<For the current data format and operation to get the current data bits
        const uint GX_PIXEL_8BIT = 0x00080000;                        ///<8 bit data image format
        const int CP_NOCLOSE_BUTTON = 0x200;                          ///<The closing button used to disable form
        int               m_nOperateID                 = 0;           ///<Camera ID
        GxBitmap m_objGxBitmap = null;                                ///<The object for show image                                                                   ///
        public ImageShowFrom(IGXDevice objIGXDevice, int nOperateID)
        {
            m_objIGXDevice = objIGXDevice;
            m_nOperateID = nOperateID + 1;

            Init();

            SetStyle(
                     ControlStyles.OptimizedDoubleBuffer
                     | ControlStyles.ResizeRedraw
                     | ControlStyles.Selectable
                     | ControlStyles.AllPaintingInWmPaint
                     | ControlStyles.UserPaint
                     | ControlStyles.SupportsTransparentBackColor,
                     true);  
            InitializeComponent();
        }

        /// <summary>
        /// Initialization operation
        /// </summary>
        void Init()
        {
            m_strSN = m_objIGXDevice.GetDeviceInfo().GetSN();
            string strValue = null;
            if (null != m_objIGXDevice)
            {
                m_nWidth = (int)m_objIGXDevice.GetRemoteFeatureControl().GetIntFeature("Width").GetValue();
                m_nHeigh = (int)m_objIGXDevice.GetRemoteFeatureControl().GetIntFeature("Height").GetValue();
                m_byMonoBuffer = new byte[m_nWidth * m_nHeigh];
                m_byColorBuffer = new byte[m_nWidth * m_nHeigh * 3];
                if (m_objIGXDevice.GetRemoteFeatureControl().IsImplemented("PixelColorFilter"))
                {
                    strValue = m_objIGXDevice.GetRemoteFeatureControl().GetEnumFeature("PixelColorFilter").GetValue();
                    if ("None" != strValue)
                    {
                        m_bIsColor = true;
                    }
                }
            }
            if (null != m_pic_ShowImage)
            {
                Bitmap bitmap = m_pic_ShowImage.Image as Bitmap;
                m_pic_ShowImage.Image = m_bitmap;
                if (bitmap != null)
                {
                    bitmap.Dispose();
                }
            }
        }

        /// <summary>
        ///Rewrite form CreateParams 
        /// </summary>
        protected override CreateParams CreateParams
        {
            get
            {
                CreateParams objCP = base.CreateParams;
                objCP.ClassStyle = objCP.ClassStyle | CP_NOCLOSE_BUTTON;
                return objCP;
            }
        }

        /// <summary>
        /// A single device to open operation
        /// </summary>
        public void OpenDevice()
        {
            //Open stream
            if (null != m_objIGXDevice)
            {
                m_objIGXStream = m_objIGXDevice.OpenStream(0);
                m_objIGXFeatureControl = m_objIGXDevice.GetRemoteFeatureControl();
                m_objIGXStreamFeatureControl = m_objIGXStream.GetFeatureControl();
                m_objGxBitmap = new GxBitmap(m_objIGXDevice, m_pic_ShowImage);

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
            }
        }

        /// <summary>
        /// Start Acquisition
        /// </summary>
        public void StartDevice()
        {
            if (null != m_objIGXStreamFeatureControl)
            {
                //Set StreamBufferHandlingMode
                m_objIGXStreamFeatureControl.GetEnumFeature("StreamBufferHandlingMode").SetValue("OldestFirst");
            }

            //Start stream channel acquisition
            if (null != m_objIGXStream)
            {
                //Register the CaptureCallback function
                m_objIGXStream.RegisterCaptureCallback(this, __OnFrameCallbackFun);
                m_objIGXStream.StartGrab();
                m_timer_UpdateFPS.Start();
            }
            //Send AcquisitionStart command
            if (null != m_objIGXFeatureControl)
            {
                m_objIGXFeatureControl.GetCommandFeature("AcquisitionStart").Execute();
            }
        }

        /// <summary>
        /// Stop Acquisition
        /// </summary>
        public void StopDevice()
        {
            m_timer_UpdateFPS.Stop();
            //Send AcquisitionStop command
            if (null != m_objIGXFeatureControl)
            {
                m_objIGXFeatureControl.GetCommandFeature("AcquisitionStop").Execute();
            }

            //Stop stream channel acquisition
            if (null != m_objIGXStream)
            {
                m_objIGXStream.StopGrab();
                //Unregister the CaptureCallback function
                m_objIGXStream.UnregisterCaptureCallback();
            }
        }

        /// <summary>
        /// Close Device
        /// </summary>
        public void CloseDevice()
        {
            try
            {
                m_timer_UpdateFPS.Stop();
                // Check whether the device has been stopped acquisition
                if (null != m_objIGXFeatureControl)
                {
                    m_objIGXFeatureControl.GetCommandFeature("AcquisitionStop").Execute();
                    m_objIGXFeatureControl = null;
                }
            }
            catch (Exception)
            {

            }
            try
            {
                // Check whether the stream has been opened
                if (null != m_objIGXStream)
                {
                    // Stop Stream acquisition
                    m_objIGXStream.StopGrab();

                    //Unregister the CaptureCallback function
                    m_objIGXStream.UnregisterCaptureCallback();

                    //Close Stream
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
                //Check whether the device has been opened
                if (null != m_objIGXDevice)
                {
                    m_objIGXDevice.Close();
                    m_objIGXDevice = null;
                    this.Close();
                }
            }
            catch (Exception)
            {

            }
        }

        /// <summary>
        /// Update image data
        /// </summary>
        /// <param name="objIBaseData">The object for image data</param>
        void __UpdateImageData(IBaseData objIBaseData)
        {
            try
            {
                GX_VALID_BIT_LIST emValidBits = GX_VALID_BIT_LIST.GX_BIT_0_7;
                if (null != objIBaseData)
                {
                    emValidBits = __GetBestValudBit(objIBaseData.GetPixelFormat());
                    if (GX_FRAME_STATUS_LIST.GX_FRAME_STATUS_SUCCESS == objIBaseData.GetStatus())
                    {
                        if (m_bIsColor)
                        {
                            IntPtr pBufferColor = objIBaseData.ConvertToRGB24(emValidBits, GX_BAYER_CONVERT_TYPE_LIST.GX_RAW2RGB_NEIGHBOUR, true);
                            Marshal.Copy(pBufferColor, m_byColorBuffer, 0, m_nWidth * m_nHeigh * 3);
                        }
                        else
                        {
                            IntPtr pBufferMono = IntPtr.Zero;
                            if (__IsPixelFormat8(objIBaseData.GetPixelFormat()))
                            {
                                pBufferMono = objIBaseData.GetBuffer();
                            }
                            else
                            {
                                pBufferMono = objIBaseData.ConvertToRaw8(emValidBits);
                            }
                            Marshal.Copy(pBufferMono, m_byMonoBuffer, 0, m_nWidth * m_nHeigh);
                        }
                    }
                }
            }
            catch (Exception)
            {
            }
        }

        /// <summary>
        /// This is a acquisition callback function that demonstrates how to get image information and show image
        /// </summary>
        /// <param name="obj">user private param</param>
        /// <param name="objIFrameData">image information</param>
        private void __OnFrameCallbackFun(object objUserParam, IFrameData objIFrameData)
        {
            try
            {
                lock (this)
                {
                    m_objCFps.IncreaseFrameNum();
                    
                    __UpdateImageData(objIFrameData);
                    m_objGxBitmap.Show(objIFrameData);
                }
            }
            catch (Exception)
            {
            }
        }

        /// <summary>
        /// Update FPS
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void m_timer_UpdateFPS_Tick(object sender, EventArgs e)
        {
            m_objCFps.UpdateFps();
            string strSN = string.Format("NO.:{0} SN:{1}",m_nOperateID.ToString(), m_strSN)  + "  FPS:";
            string strText = strSN + m_objCFps.GetFps().ToString("F2");
            this.Text = strText;
        }


        /// <summary>
        /// Check whether the PixelFormat is 8 bits
        /// </summary>
        /// <param name="emPixelFormatEntry">image format</param>
        /// <returns>true:  8 bit  false: not 8 bit</returns>
        private bool __IsPixelFormat8(GX_PIXEL_FORMAT_ENTRY emPixelFormatEntry)
        {
            bool bIsPixelFormat8 = false;
            uint uiPixelFormatEntry = (uint)emPixelFormatEntry;
            if ((uiPixelFormatEntry & PIXEL_FORMATE_BIT) == GX_PIXEL_8BIT)
            {
                bIsPixelFormat8 = true;
            }
            return bIsPixelFormat8;
        }

        /// <summary>
        /// Get the best 8 bit by GX_PIXEL_FORMAT_ENTRY
        /// </summary>
        /// <param name="em">image format</param>
        /// <returns>best bit bit</returns>
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
                        //There is no such data format to be upgraded
                        break;
                    }
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_MONO16:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GR16:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_RG16:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GB16:
                case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_BG16:
                    {
                        //There is no such data format to be upgraded
                        break;
                    }
                default:
                    break;
            }
            return emValidBits;
        }

        /// <summary>
        /// </summary>
        /// <param name="e"></param>
        protected override void OnResize(EventArgs e)
        {
            base.OnResizeEnd(e);
            m_pic_ShowImage.Width = this.Width;
            m_pic_ShowImage.Height = this.Height;

        }

        private void m_pic_ShowImage_Click(object sender, EventArgs e)
        {

        }

    }
}
