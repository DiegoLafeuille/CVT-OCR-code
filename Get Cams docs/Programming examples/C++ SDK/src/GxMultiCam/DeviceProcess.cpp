//------------------------------------------------------------
/**
\file    DeviceProcess.cpp 
\brief   Device Process 
*/
//------------------------------------------------------------

#include "StdAfx.h"
#include "DeviceProcess.h"

CDeviceProcess::CDeviceProcess(void)
{
	m_pBitmap                   = NULL;
	m_pSampleCaptureEventHandle = NULL;
	m_bIsOpen                   = false;
	m_bIsSnap                   = false;
	m_bShowDeviceFPS            = true;
	m_nDeviceIndex              = 0;
	m_strDeviceSN               = "";
	m_pSampleCaptureEventHandle = new CSampleCaptureEventHandler();
	m_pRecordDeviceSNFPS        = new char[1024];
}

CDeviceProcess::~CDeviceProcess(void)
{
	if (m_pSampleCaptureEventHandle != NULL)
	{
		delete m_pSampleCaptureEventHandle;
		m_pSampleCaptureEventHandle = NULL;
	}

	if (m_pRecordDeviceSNFPS != NULL)
	{
		delete[] m_pRecordDeviceSNFPS;
		m_pRecordDeviceSNFPS = NULL;
	}
}

//------------------------------------------------------------
/**
\brief   Open Device 
\param   strDeviceSN   [in]         Device serial number
\param   m_pWnd        [in]         the handle for window
\param   nDeviceIndex  [in]         Device index
\return  void
*/
//------------------------------------------------------------
void CDeviceProcess::OpenDevice(gxstring strDeviceSN, CWnd* m_pWnd,int nDeviceIndex)
{
	bool bIsDeviceOpen   = false;      ///< The flag indicate whether the device has been opened
	bool bIsStreamOpen   = false;      ///< The flag indicate whether the stream has been opened
	try
	{
		if (m_bIsOpen || m_pWnd == NULL || strDeviceSN == "")
		{
			throw std::exception("Invalid parameter!");
		}

		m_strDeviceSN = strDeviceSN;
		m_nDeviceIndex  = nDeviceIndex;

		// Open device
		m_objDevicePtr = IGXFactory::GetInstance().OpenDeviceBySN(strDeviceSN,GX_ACCESS_EXCLUSIVE);
		bIsDeviceOpen = true;

		// Get the feature control
		m_objFeatureControlPtr = m_objDevicePtr->GetRemoteFeatureControl();

		// Set the continuous frame acquisition mode
		m_objFeatureControlPtr->GetEnumFeature("AcquisitionMode")->SetValue("Continuous");

		// Set the TriggerMode off
		m_objFeatureControlPtr->GetEnumFeature("TriggerMode")->SetValue("Off");

		// Get the number of the device stream 
		int nCount = m_objDevicePtr->GetStreamCount();

		if (nCount > 0)
		{
			m_objStreamPtr = m_objDevicePtr->OpenStream(0);
			m_objStreamFeatureControlPtr = m_objStreamPtr->GetFeatureControl();
			bIsStreamOpen = true;
		}
		else
		{
			throw std::exception("No stream found!");
		}

        //It is recommended that the user set the camera's stream channel packet length value 
        //according to the current network environment after turning on 
        //the network camera to improve the collection performance of the network camera. 
        //For the setting method, refer to the following code.
        GX_DEVICE_CLASS_LIST objDeviceClass = m_objDevicePtr->GetDeviceInfo().GetDeviceClass();
        if(GX_DEVICE_CLASS_GEV == objDeviceClass)
        {
            // Determine whether the device supports the stream channel packet function.
            if(true == m_objFeatureControlPtr->IsImplemented("GevSCPSPacketSize"))
            {
                // Get the optimal packet length value of the current network environment
                int nPacketSize = m_objStreamPtr->GetOptimalPacketSize();
                // Set the optimal packet length value to the stream channel packet length of the current device.
                m_objFeatureControlPtr->GetIntFeature("GevSCPSPacketSize")->SetValue(nPacketSize);
            } 
        }

		m_bIsOpen = true;

		if (m_pBitmap != NULL)
		{
			delete m_pBitmap;
			m_pBitmap = NULL;
		}
		m_pBitmap = new CGXBitmap(m_objDevicePtr,m_pWnd);
	}
	catch (CGalaxyException& e)
	{

		// Check whether the device stream has been opened
		if (bIsStreamOpen)
		{
			m_objStreamPtr->Close();
		} 

		// Check whether the device has been opened
		if (bIsDeviceOpen)
		{
			m_objDevicePtr->Close();
		}

		if (m_pBitmap != NULL)
		{
			delete m_pBitmap;
			m_pBitmap = NULL;
		}

		m_bIsOpen = false;

		throw e;

	}
	catch (std::exception& e)
	{
		// Check whether the device stream has been opened
		if (bIsStreamOpen)
		{
			m_objStreamPtr->Close();
		} 

		// Check whether the device has been opened
		if (bIsDeviceOpen)
		{
			m_objDevicePtr->Close();
		}

		if (m_pBitmap != NULL)
		{
			delete m_pBitmap;
			m_pBitmap = NULL;
		}

		m_bIsOpen = false;

		throw e;
	}

}
//------------------------------------------------------------
/**
\brief   Close Device 

\return  void
*/
//------------------------------------------------------------
void CDeviceProcess::CloseDevice()
{
	if (!m_bIsOpen)
	{
		return;
	}

	try
	{
		// Check whether the device has been stopped acquisition
		if (m_bIsSnap)
		{

			// Send AcquisitionStop command
			m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();

			// Stop Stream acquisition
			m_objStreamPtr->StopGrab();

			// Unregister the CaptureCallback function
			m_objStreamPtr->UnregisterCaptureCallback();
		}
	}
	catch (CGalaxyException)
	{
		//do noting
	}

	try
	{
		// Close stream
		m_objStreamPtr->Close();
	}
	catch (CGalaxyException)
	{
		//do noting
	}

	try
	{
		// Close device
		m_objDevicePtr->Close();
	}
	catch (CGalaxyException)
	{
		//do noting
	}

	m_bIsSnap = false;
	m_bIsOpen = false;

	if (m_pBitmap != NULL)
	{
		delete m_pBitmap;
		m_pBitmap = NULL;
	}

}
//------------------------------------------------------------
/**
\brief   Start Snap 

\return  void
*/
//------------------------------------------------------------
void CDeviceProcess::StartSnap()
{
	// Check whether the device has been opened
	if (!m_bIsOpen)
	{
		return;
	}

	try
	{
		// Set StreamBufferHandlingMode
		m_objStreamFeatureControlPtr->GetEnumFeature("StreamBufferHandlingMode")->SetValue("OldestFirst");

		// Register the CaptureCallback function
		m_objStreamPtr->RegisterCaptureCallback(m_pSampleCaptureEventHandle,this);
	}
	catch (CGalaxyException& e)
	{
		throw e;
	}

	try
	{

		// Start stream channel acquisition
		m_objStreamPtr->StartGrab();
	}
	catch (CGalaxyException& e)
	{
		// Unregister the CaptureCallback function
		m_objStreamPtr->UnregisterCaptureCallback();
		throw e;
	}

	try
	{
		// Send AcquisitionStart command
		m_objFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();

		m_bIsSnap = true;
	}
	catch (CGalaxyException& e)
	{
		// Stop stream channel acquisition
		m_objStreamPtr->StopGrab();

		// Unregister the CaptureCallback function
		m_objStreamPtr->UnregisterCaptureCallback();
		throw e;
	}
	catch (std::exception& e)
	{
		// Stop stream channel acquisition
		m_objStreamPtr->StopGrab();

		// Unregister the CaptureCallback function
		m_objStreamPtr->UnregisterCaptureCallback();
		throw e;
	}

}

//------------------------------------------------------------
/**
\brief   Stop Snap 

\return  void
*/
//------------------------------------------------------------
void CDeviceProcess::StopSnap()
{
	// Check whether the device has been opened
	if (!m_bIsOpen || !m_bIsSnap)
	{
		return;
	}

	try
	{
		// Send AcquisitionStop command
		m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();

		// Stop stream channel acquisition
		m_objStreamPtr->StopGrab();

		// Unregister the CaptureCallback function
		m_objStreamPtr->UnregisterCaptureCallback();

		// frame rate is set 0 
		m_objfps.Reset();

		m_bIsSnap = false;
	}
	catch (CGalaxyException& e)
	{
		throw e;

	}
	catch (std::exception& e)
	{
		throw e;

	}
}

//------------------------------------------------------------
/**
\brief   Get Device Open Flag 

\return  void
*/
//------------------------------------------------------------
bool CDeviceProcess::IsOpen() const
{
	return m_bIsOpen;
}

//------------------------------------------------------------
/**
\brief   Get Device Snap Flag 

\return  void
*/
//------------------------------------------------------------
bool CDeviceProcess::IsSnap() const
{
	return m_bIsSnap;
}

//------------------------------------------------------------
/**
\brief   Set Show frame Frequency Flag

\return  void
*/
//------------------------------------------------------------
void CDeviceProcess::SetShowFPS(bool bIsShowFps)
{
	m_bShowDeviceFPS = bIsShowFps;
}

//------------------------------------------------------------
/**
\brief   Get Show frame Frequency Flag

\return  void
*/
//------------------------------------------------------------
bool CDeviceProcess::GetShowFPS()
{
	return m_bShowDeviceFPS;
}

//------------------------------------------------------------
/**
\brief   Refresh Device SN and FPS

\return  void
*/
//------------------------------------------------------------
void CDeviceProcess::RefreshDeviceSNFPS()
{
	double dFps = 0;        ///< frame rate

			// Update frame rate
			m_objfps.UpdateFps();

		// Get the latest frame rate
		dFps = m_objfps.GetFps();

	sprintf_s(m_pRecordDeviceSNFPS, 1024,
			"Serial number:%d SN:%s FPS:%.2f",
			m_nDeviceIndex,
			m_strDeviceSN.c_str(),
			dFps);
}

//------------------------------------------------------------
/**
\brief   Show Image
\param   objImageDataPointer [in]       Image process

\return  void
*/
//------------------------------------------------------------
void CDeviceProcess::__ShowPicture(CImageDataPointer& objImageDataPointer)
{
	if (m_bShowDeviceFPS)
	{
		// Increase the number of frames
		m_objfps.IncreaseFrameNum();

		// Show Image and frame rate
		m_pBitmap->Show(objImageDataPointer,m_pRecordDeviceSNFPS);

	}
	else
	{
		// Show Image
		m_pBitmap->Show(objImageDataPointer);
	}
}

