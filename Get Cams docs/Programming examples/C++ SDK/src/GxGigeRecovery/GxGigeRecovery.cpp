//----------------------------------------------------------------------------------
/**
\file    GxGigeRecovery.cpp 
\brief   CGxGigeRecovery class declaration implementation
*/
//----------------------------------------------------------------------------------
#include "stdafx.h"
#include "stdio.h"
#include "GxGigeRecovery.h"

using namespace std;

//---------------------------------------------------------------------------------
/**
\brief   The construction function

\return   void
*/
//----------------------------------------------------------------------------------
CGigeRecovery::CGigeRecovery()
{
	try
	{
		m_hCB              = NULL;
		m_strSavePath      = "";
		m_strMACAddress    = "";
		m_bIsOffline       = false;
		m_bIsOpen          = false;
		m_bIsSnap          = false;
		m_pDeviceOfflineEventHandle = new CSampleDeviceOfflineEventHandle();

		// Before using any GxIAPICPP methods, the GxIAPICPP must be initialized.
		IGXFactory::GetInstance().Init();

		// Get current file path to save the configuration param
		char chModulePath[512] = {0};
		string strPath       = "";
		::GetModuleFileName(NULL,(LPTSTR)chModulePath,512);
		strPath = chModulePath;
		strPath = strPath.substr(0,strPath.rfind('\\')+1);
        strPath = strPath + "ConfigFile.ini";
		m_strSavePath  = strPath.c_str();
	}
	catch (CGalaxyException &e)
	{
		if (m_pDeviceOfflineEventHandle != NULL)
		{
			delete m_pDeviceOfflineEventHandle;
		}
		cout<<"<"<<e.GetErrorCode()<<">"<<"<"<<e.what()<<">"<<endl;		
	}
	catch (std::exception &e)
	{
		if (m_pDeviceOfflineEventHandle != NULL)
		{
			delete m_pDeviceOfflineEventHandle;
		}
		cout<<"<"<<e.what()<<">"<<endl;		
	}

}

//---------------------------------------------------------------------------------
/**
\brief   The destruction function 

\return   void
*/
//----------------------------------------------------------------------------------
CGigeRecovery::~CGigeRecovery()
{
	try
	{
		// Release GxIAPICPP
		IGXFactory::GetInstance().Uninit();
	}
	catch (CGalaxyException &e)
	{
		cout<<"<"<<e.GetErrorCode()<<">"<<"<"<<e.what()<<">"<<endl;			
	}
	catch (std::exception &e)
	{
		cout<<"<"<<e.what()<<">"<<endl;			
	}

	if (m_pDeviceOfflineEventHandle != NULL)
	{
		delete m_pDeviceOfflineEventHandle;
		m_pDeviceOfflineEventHandle = NULL;
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to enumerate all camera devices and initialize device.

\return   true: success     false: fail
*/
//----------------------------------------------------------------------------------
bool CGigeRecovery::__OnOpenDevice() throw()
{
	bool bIsDeviceOpen = false;              ///< The flag indicates whether the device has been opened
	bool bIsStreamOpen = false;              ///< The flag indicates whether the stream has been opened

	try
	{
		cout<<"====================CGigeRecovery::__OnOpenDevice()===================="<<endl;

		 // Enumerate all camera devices
		GxIAPICPP::gxdeviceinfo_vector vectorDeviceInfo;
		IGXFactory::GetInstance().UpdateDeviceList(1000,vectorDeviceInfo);

		// Check if found any device
		if (vectorDeviceInfo.size() <= 0)
		{
			cout<<"No found device!"<<endl;
			return m_bIsOpen;
		}

		// Check whether the device is Gige
		if (vectorDeviceInfo[0].GetDeviceClass() != GX_DEVICE_CLASS_GEV)
		{
			cout<<"The device is not GEV"<<endl;
			return m_bIsOpen;
		}        	
        
		// Open device by MAC address
		m_strMACAddress = vectorDeviceInfo[0].GetMAC();	
		cout<<"<Open device by MAC:"<<m_strMACAddress<<endl;
		m_objDevicePtr          = IGXFactory::GetInstance().OpenDeviceByMAC(m_strMACAddress,GX_ACCESS_EXCLUSIVE);
		bIsDeviceOpen = true;
		m_objFeatureControlPtr  = m_objDevicePtr->GetRemoteFeatureControl();
		int32_t nStreamCount    = m_objDevicePtr->GetStreamCount();

		// Check if found any stream
		if (nStreamCount > 0)
		{
			// Open Stream
			m_objStreamPtr = m_objDevicePtr->OpenStream(0);
			m_objStreamFeatureControlPtr = m_objStreamPtr->GetFeatureControl();
			bIsStreamOpen = true;
		}
		else
		{
			throw exception("No found stream!");
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

		// Initialize the device parameters
		cout<<"<Initialize the device parameters>"<<endl;
		__InitParam();

		// Export the parameter config file 
		cout<<"<Export config file>"<<endl;
		m_objDevicePtr->ExportConfigFile(m_strSavePath);

		// Register the OffLine callback function
		cout<<"<Register device Offline callback>"<<endl;
		m_hCB = m_objDevicePtr->RegisterDeviceOfflineCallback(m_pDeviceOfflineEventHandle,this);
		m_bIsOpen = true;
		return m_bIsOpen;

	}
	catch (CGalaxyException &e)
	{
		// Check whether the stream has been closed
		if (bIsStreamOpen)
		{
			m_objStreamPtr->Close();
		}

		// Check whether the device has been closed
		if (bIsDeviceOpen)
		{
			m_objDevicePtr->Close();
		}
		cout<<"<"<<e.GetErrorCode()<<">"<<"<"<<e.what()<<">"<<endl;
		return m_bIsOpen;		
	}
	catch (std::exception &e)
	{
		// Check whether the stream has been closed
		if (bIsStreamOpen)
		{
			m_objStreamPtr->Close();
		}

		// Check whether the device has been closed
		if (bIsDeviceOpen)
		{
			m_objDevicePtr->Close();
		}
		cout<<"<"<<e.what()<<">"<<endl;
		return m_bIsOpen;		
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize device

\return   void
*/
//----------------------------------------------------------------------------------
void CGigeRecovery::__InitParam()
{
	// Set the continuous frame acquisition mode
	m_objFeatureControlPtr->GetEnumFeature("AcquisitionMode")->SetValue("Continuous");

	// Close the TriggerMode
	m_objFeatureControlPtr->GetEnumFeature("TriggerMode")->SetValue("Off");

	//Set heartbeat timeout for 1s
	//If you build this sample in debug mode and run it using a GigE camera device,the device will be set heartbeat timeout to 5 minutes
	//This is done to prevent the camera's heartbeat timeout from affecting the program's debugging and single stepping of execution, This also means that the camera can not be disconnected for 5 minutes unless the camera is powered off and on 
	//To work around this, the heartbeat timeout will be set to 1000 ms 
	m_objFeatureControlPtr->GetIntFeature("GevHeartbeatTimeout")->SetValue(1000);

	//Set StreamBufferHandlingMode
	m_objStreamFeatureControlPtr->GetEnumFeature("StreamBufferHandlingMode")->SetValue("OldestFirst");
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to start acquisition

\return   true: success, false: fail
*/
//----------------------------------------------------------------------------------
bool CGigeRecovery::__OnStartSnap() throw()
{
	try
	{
		cout<<"====================CGigeRecovery::__OnStartSnap()===================="<<endl;

		// Start stream channel acquisition
		m_objStreamPtr->StartGrab();

		// Send acquisitionStart command
		cout<<"<Send start snap command to device>"<<endl;
		m_objFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();
		m_bIsSnap = true;
		return m_bIsSnap;
	}
	catch (CGalaxyException &e)
	{
		cout<<"<"<<e.GetErrorCode()<<">"<<"<"<<e.what()<<">"<<endl;
		return m_bIsSnap;		
	}
	catch (std::exception &e)
	{
		cout<<"<"<<e.what()<<">"<<endl;
		return m_bIsSnap;		
	}

}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to stop acquisition

\return   true: success, false: fail
*/
//----------------------------------------------------------------------------------
bool CGigeRecovery::__OnStopSnap() throw()
{
	try
	{
		cout<<"====================CGigeRecovery::__OnStopSnap()===================="<<endl;

		// Send acquisitionStop command
		cout<<"<Send stop snap command to device>"<<endl;
		m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();

		// Stop stream channel acquisition
		m_objStreamPtr->StopGrab();
		m_bIsSnap = false;
		return m_bIsSnap;
	}
	catch (CGalaxyException &e)
	{
		cout<<"<"<<e.GetErrorCode()<<">"<<"<"<<e.what()<<">"<<endl;
		return m_bIsSnap;		
	}
	catch (std::exception &e)
	{
		cout<<"<"<<e.what()<<">"<<endl;
		return m_bIsSnap;		
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to close device

\return   true: success, false: fail
*/
//----------------------------------------------------------------------------------
bool CGigeRecovery::__OnCloseDevice() throw()
{
	try
	{
		cout<<"====================CGigeRecovery::__OnCloseDevice()===================="<<endl;

		// Unregister the device OffLine callback function 
		cout<<"<Unregister device Offline callback>"<<endl;
		m_objDevicePtr->UnregisterDeviceOfflineCallback(m_hCB);
		m_objStreamPtr->Close();

		// Close device
		cout<<"<Close device>"<<endl;
		m_objDevicePtr->Close();
		m_bIsOpen = false;
		return m_bIsOpen;
	}
	catch (CGalaxyException &e)
	{
		cout<<"<"<<e.GetErrorCode()<<">"<<"<"<<e.what()<<">"<<endl;
		return m_bIsOpen;		
	}
	catch (std::exception &e)
	{
		cout<<"<"<<e.what()<<">"<<endl;
		return m_bIsOpen;		
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to acquire image continuously

\return   void
*/
//----------------------------------------------------------------------------------
void CGigeRecovery::__Acquisition() throw()
{
	cout<<"====================CGigeRecovery::__Acquisition()===================="<<endl;;
	cout<<"<Press any key to stop acquisition>"<<endl;

	// Check whether the keyboard is pressed
	while(!_kbhit())
	{
		try
		{
			if (m_bIsOffline)        // The device is OffLine
			{
				// Deal with offline
				__ProcessOffline();

				// Device recovery
				__Recovery();
			}
			else                    // The device is OnLine 
			{
				m_objImageDataPtr = m_objStreamPtr->GetImage(500);
				printf("Successfully get Image\n");
			}
		}
		catch(CGalaxyException &e)
		{
			cout<<"<"<<e.GetErrorCode()<<">"<<"<"<<e.what()<<">"<<endl;
		}
		catch(std::exception &e)
		{
			cout<<"<"<<e.what()<<">"<<endl;
		}

	}
	_getch();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to deal with the OffLine event

\return   void
*/
//----------------------------------------------------------------------------------
void CGigeRecovery::__ProcessOffline() throw()
{
	try
	{
		printf("**********************Process Offline**********************\r");

		// If the device is acquiring image then stop it.
		if (m_bIsSnap)
		{
			cout<<"\n<Send stop snap command to device>"<<endl;
			m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
		}
	}
	catch(CGalaxyException &e)
	{
		cout<<"<"<<e.GetErrorCode()<<">"<<"<"<<e.what()<<">"<<endl;
	}
	catch(std::exception &e)
	{
		cout<<"<"<<e.what()<<">"<<endl;
	}

	try
	{
		// If the device is acquiring image then stop it.
		if (m_bIsSnap)
		{
			m_objStreamPtr->StopGrab();
			m_bIsSnap = false;
		}
	}
	catch(CGalaxyException)
	{
		//do noting
	}
	catch(std::exception)
	{
		//do noting
	}

	try
	{
		// If the device is opened then close it.
		if (m_bIsOpen)
		{
			// Unregister the OffLine callback function
			cout<<"<Unregister device Offline callback>"<<endl;
			m_objDevicePtr->UnregisterDeviceOfflineCallback(m_hCB);

			// Close stream and device
			cout<<"<Close device>"<<endl;
			m_objStreamPtr->Close();
			m_objDevicePtr->Close();
			m_bIsOpen = false;

		}
	}
	catch(CGalaxyException)
	{
		//do noting
	}
	catch(std::exception)
	{
		//do noting
	}

}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to recovery the device from OffLine status

\return   void
*/
//----------------------------------------------------------------------------------
void CGigeRecovery::__Recovery() throw()
{
	bool bIsDeviceOpen = false;              ///< The flag indicates whether the device has been opened
	bool bIsStreamOpen = false;              ///< The flag indicates whether the stream has been opened
	try
	{
        printf("**********************Recovery**********************\r");

		 // Enumerate all camera devices
		GxIAPICPP::gxdeviceinfo_vector vectorDeviceInfo;
		IGXFactory::GetInstance().UpdateDeviceList(1000,vectorDeviceInfo);

		// Check if found any device
		if (vectorDeviceInfo.size() <= 0)
		{
			return;
		}

		// Open device by MAC address
		cout<<"\n<Open device by MAC:"<<m_strMACAddress<<endl;
		m_objDevicePtr          = IGXFactory::GetInstance().OpenDeviceByMAC(m_strMACAddress,GX_ACCESS_EXCLUSIVE);
		bIsDeviceOpen = true;
		m_objFeatureControlPtr  = m_objDevicePtr->GetRemoteFeatureControl();

		// Check if found any stream
		int32_t nStreamCount    = m_objDevicePtr->GetStreamCount();
		if (nStreamCount > 0)
		{
			m_objStreamPtr = m_objDevicePtr->OpenStream(0);
			bIsStreamOpen = true;
		}
		else
		{
			throw exception("No found stream!");
		}

		// Import config file
		cout<<"<Import config file>"<<endl;
		m_objDevicePtr->ImportConfigFile(m_strSavePath);

		// Register the OffLine callback function
		cout<<"<Register device Offline callback>"<<endl;
		m_hCB = m_objDevicePtr->RegisterDeviceOfflineCallback(m_pDeviceOfflineEventHandle,this);
		m_bIsOpen = true;

		// Send acquisitionStart command
		cout<<"<Send start snap command to device>"<<endl;
		m_objStreamPtr->StartGrab();
		m_objFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();
		m_bIsSnap    = true;
		m_bIsOffline = false;
	}
	catch(CGalaxyException &e)
	{
		// Check whether the stream has been opend
		if (bIsStreamOpen)
		{
			m_objStreamPtr->Close();
		}

		//Check whether the device has been opend
		if (bIsDeviceOpen)
		{
			m_objDevicePtr->Close();
		}

		cout<<"<"<<e.GetErrorCode()<<">"<<"<"<<e.what()<<">"<<endl;
	}
	catch(std::exception &e)
	{
		//Check whether the stream has been opend
		if (bIsStreamOpen)
		{
			m_objStreamPtr->Close();
		}

		//Check whether the device has been opend
		if (bIsDeviceOpen)
		{
			m_objDevicePtr->Close();
		}

		cout<<"<"<<e.what()<<">"<<endl;
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to recovery from offLine status

\return   void
*/
//----------------------------------------------------------------------------------
void CGigeRecovery::GigeRecovery()
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
			// Close device
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
	catch (std::exception &e)
	{
		cout<<"<"<<e.what()<<">"<<endl;
	}

}

