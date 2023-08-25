using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using GxIAPINET;
using GxIAPINET.Sample.Common;

namespace GxMultiCam
{
    public class CCamerInfo
    {
        public bool m_bIsColorFilter = false;                                                                       ///<The flag indicates whether the device is color
        public bool m_bIsOpen = false;	                                                                            ///<The flag indicates whether the device has been opened
        public bool m_bIsSnap = false;		                                                                        ///<The flag indicates whether the device is starting acquisition
        public bool m_bWhiteAuto = false;                                                                           ///<The flag indicates whether the device has been support white balance
        public bool m_bAcqSpeedLevel = false;                                                                       ///<The flag indicates whether the device has been support acquisition speed level
        public bool m_bWhiteAutoSelectedIndex = true;                                                               ///<The flag indicates whether the combobox has been changed
        public double                m_dFps                       = 0.0;			                                ///<Frame rate
        public IGXDevice             m_objIGXDevice               = null;                                           ///<The object for device
        public IGXStream             m_objIGXStream               = null;                                           ///<The object for stream
        public IGXFeatureControl     m_objIGXFeatureControl       = null;                                           ///<The object for feature control
        public string                m_strBalanceWhiteAutoValue   = "Off";                                          ///<current value of white balance auto
        public string                m_strDisplayName             = "";                                             ///<Device Name
        public string m_strSN = "";                                                                                 ///<Device serial name
        public GX_DEVICE_CLASS_LIST  m_emDeviceType               = GX_DEVICE_CLASS_LIST.GX_DEVICE_CLASS_UNKNOWN;   ///<Device class  
        public ImageShowFrom         m_objImageShowFrom           = null;                                           ///<Image show
    }
}
