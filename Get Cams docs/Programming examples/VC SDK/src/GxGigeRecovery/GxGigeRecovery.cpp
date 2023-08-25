//----------------------------------------------------------------------------------
/**
\file    GxGigeRecovery.cpp 
\brief   Gige recovery function
\version v1.0.1403.9211
\date    2014-3-21
*/
//----------------------------------------------------------------------------------
#include "stdafx.h"
#include "GxIAPI.h"
#include <conio.h>
#include <iostream>
#include <string>
using namespace std;
#include <TCHAR.H>

extern "C" WINBASEAPI HWND WINAPI GetConsoleWindow();
//----------------------------------------------------------------------------------
/** 
\brief   This is a class for controlling device 
*/
//----------------------------------------------------------------------------------
class CGigeRecovery
{
public:
	CGigeRecovery();
    virtual ~CGigeRecovery();

	/// The external interface function for OffLine recovery 
	void GigeRecovery();

private:

	bool __OnOpenDevice();

	bool __InitParam();
	
	bool __OnStartSnap();
	
	bool __OnStopSnap();

	bool __OnCloseDevice();
	
	void __Acquisition();

	void __ProcessOffline();

	void __Recovery();

	void __GetErrorString(GX_STATUS emErrorStatus);

	static void __stdcall __OnDeviceOfflineCallbackFun(void* pUserParam);

private:
	GX_DEV_HANDLE             m_hDevice;                              ///< The handle for device             
	GX_EVENT_CALLBACK_HANDLE  m_hCB;                                  ///< The handle for OffLine callback event  
	string                    m_strPath;                              ///< The path for saving current files  
	string                    m_strSavePath;                          ///< The path for saving device configuration parameters
	char                      m_chModulePath[100];                    ///< Get current working path 
	char                      m_chMACAddress[GX_INFO_LENGTH_32_BYTE]; ///< The MAC address of device 
	GX_FRAME_DATA             m_stFrameData;                          ///< The structure for acquired image data   
	bool                      m_bIsOffLine;                           ///< The flag indicates whether the device is OffLine
	bool                      m_bIsOpen;                              ///< The flag indicates whether the first device has been opened
	bool                      m_bIsSanp;                              ///< The flag indicates whether AcquisitionStart command has been send 
};


int main(int argc, char* argv[])
{
	// Call the CGigeRecovery class for demonstrating how to recovery from the offLine status   
	CGigeRecovery objCtrDevice;
	objCtrDevice.GigeRecovery();

	// Press any key to exit the console application
	printf("\n<press any key to exit>\n");
	while (!_kbhit());

	return 0;
}

//---------------------------------------------------------------------------------
/**
\brief    The construction function

 \return  void 
*/
//----------------------------------------------------------------------------------
CGigeRecovery::CGigeRecovery()
{ 
	m_hDevice             = NULL;              // The handle for device   
	m_hCB                 = NULL;              // The handle for OffLine callback event
	m_stFrameData.pImgBuf = NULL;              // The buffer allocated for acquiring image 
	m_strPath             = "";                // Get current working path 
	m_strSavePath         = "";                // The path for saving current files
	m_bIsOffLine          = false;             // The flag indicates whether the device is OffLine
	m_bIsOpen             = false;             // The flag indicates whether the first device has been opened
	m_bIsSanp             = false;             // The flag indicates whether AcquisitionStart command has been send 

	// Before using any GxIAPI methods, the GxIAPI must be initialized.   
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	emStatus = GXInitLib();
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		exit(0);
	}

	// Get current file path to save the configuration files
	GetModuleFileName(NULL, m_chModulePath, sizeof(m_chModulePath));
	(_tcsrchr(m_chModulePath, _T('\\')))[1] = 0;
	m_strPath = m_chModulePath;
	m_strSavePath = m_strPath + "ConfigFile.ini";	
}

//---------------------------------------------------------------------------------
/**
\brief    The destruction function           
 
\return   void
*/
//----------------------------------------------------------------------------------
CGigeRecovery::~CGigeRecovery()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Release GxIAPI   
	emStatus = GXCloseLib();
}

//---------------------------------------------------------------------------------
/**
\brief    This function demonstrates how to enumerate all camera devices and initialize device.

\return   true: Success, false: Fail
*/
//----------------------------------------------------------------------------------
bool CGigeRecovery::__OnOpenDevice() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS         emStatus   = GX_STATUS_SUCCESS;   
	uint32_t          nDeviceNum = 0;        
	bool              bReturn    = false;    
    GX_DEVICE_IP_INFO stDeviceIPInfo; 
	GX_DEVICE_BASE_INFO stDeviceBaseInfo;
	size_t nDeviceBaseInfoSize = 0;
	GX_OPEN_PARAM     stOpenParam;          

    // Enumerate all camera devices
	printf("====================CGigeRecovery::__OnOpenDevice()====================\n");
	emStatus = GXUpdateDeviceList(&nDeviceNum, 1000);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return false;
	}
		
	// Check if found any device
	if (nDeviceNum <= 0)
	{
		printf("<NO device>\n");
 		return false;
	}

	//Get Device Base Information
	nDeviceBaseInfoSize = sizeof(stDeviceBaseInfo);
	emStatus = GXGetAllDeviceBaseInfo(&stDeviceBaseInfo, &nDeviceBaseInfoSize);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return false;
	}

	//Check whether the device is Gige
	if (stDeviceBaseInfo.deviceClass != GX_DEVICE_CLASS_GEV)
	{
		printf("<The device is not GEV>");
		return false;
	}

	//Get the IP information of the first device
	emStatus = GXGetDeviceIPInfo(1, &stDeviceIPInfo);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return false;
	}
	memcpy(m_chMACAddress, stDeviceIPInfo.szMAC, GX_INFO_LENGTH_32_BYTE);
	
	//Open device by MAC address
	stOpenParam.accessMode = GX_ACCESS_EXCLUSIVE;
	stOpenParam.openMode   = GX_OPEN_MAC;
	stOpenParam.pszContent = stDeviceIPInfo.szMAC;
	printf("<Open device by MAC: %s>\n", stDeviceIPInfo.szMAC);
	emStatus = GXOpenDevice(&stOpenParam, &m_hDevice);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return false;
	}

    // It is suggested that users set the camera's packet size according to the current network environment after opening the network camera.
    // To improve the acquisition performance of network cameras, the following code is used for reference.
    {
        bool	 bImplementPacketSize = false;
        uint32_t unPacketSize		  = 0;

        emStatus = GXIsImplemented(m_hDevice, GX_INT_GEV_PACKETSIZE, &bImplementPacketSize);
        if (emStatus != GX_STATUS_SUCCESS)
        {
            __GetErrorString(emStatus);
            return false;
        }

        if (bImplementPacketSize)
        {
            emStatus = GXGetOptimalPacketSize(m_hDevice,&unPacketSize);
            if (emStatus != GX_STATUS_SUCCESS)
            {
                __GetErrorString(emStatus);
                return false;
            }

            emStatus = GXSetInt(m_hDevice, GX_INT_GEV_PACKETSIZE, unPacketSize);
            if (emStatus != GX_STATUS_SUCCESS)
            {
                __GetErrorString(emStatus);
                return false;
            }
        }
    }
	
	printf("<Initialize the device parameters>\n");
	bReturn = __InitParam();
	if (!bReturn)
	{
		return false;
	}

	// Export the parameter config file 
	printf("<Export config file>\n");
	emStatus = GXExportConfigFile(m_hDevice, m_strSavePath.c_str());
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return false;
	}

	// Register the OffLine callback function
	printf("<Register device Offline callback>\n");
	emStatus = GXRegisterDeviceOfflineCallback(m_hDevice, this, __OnDeviceOfflineCallbackFun, &m_hCB);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return false;
	}

	m_bIsOpen = true;	
	return true;
}

//---------------------------------------------------------------------------------
/**
\brief    This function demonstrates how to initialize device


\return   true: Success, false: Fail
*/
//----------------------------------------------------------------------------------
bool CGigeRecovery::__InitParam()
{
	GX_STATUS emStatus     = GX_STATUS_SUCCESS;  
	int64_t   nPayLoadSize = 0;     
	
	// Set the continuous frame acquisition mode
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return false;
	}

	// Close the TriggerMode
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_OFF);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return false;
	}

	//Set StreamBufferHandlingMode
	emStatus = GXSetEnum(m_hDevice, GX_DS_ENUM_STREAM_BUFFER_HANDLING_MODE, GX_DS_STREAM_BUFFER_HANDLING_MODE_OLDEST_FIRST);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return false;
	}

	//Set heartbeat timeout for 1s
	//If you build this sample in debug mode and run it using a GigE camera device,the device will be set heartbeat timeout to 5 minutes
	//This is done to prevent the camera's heartbeat timeout from affecting the program's debugging and single stepping of execution, This also means that the camera can not be disconnected for 5 minutes unless the camera is powered off and on 
	//To work around this, the heartbeat timeout will be set to 1000 ms 
	emStatus = GXSetInt(m_hDevice, GX_INT_GEV_HEARTBEAT_TIMEOUT, 1000);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return false;
	}
	
	// Get the PayloadSize of the image acquired from the camera
    emStatus = GXGetInt(m_hDevice, GX_INT_PAYLOAD_SIZE, &nPayLoadSize);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return false;
	}
	m_stFrameData.pImgBuf = new BYTE[(size_t)nPayLoadSize];
	if (m_stFrameData.pImgBuf == NULL)
	{
		printf("<Failed to allocate memory>\n");
		return false;
	}

	return true;
}

//---------------------------------------------------------------------------------
/**
\brief   This is a callback function that demonstrates how to deal with OffLine event
\param   pUserParam  

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CGigeRecovery::__OnDeviceOfflineCallbackFun(void* pUserParam)
{
	CGigeRecovery *pCtrDevice = (CGigeRecovery *)pUserParam;

	pCtrDevice->m_bIsOffLine = true;
	printf("**********************Device offline**********************\n");
}

//---------------------------------------------------------------------------------
/**
\brief    This function demonstrates how to start acquisition                         

\return   true: Success, false: Fail
*/
//----------------------------------------------------------------------------------
bool CGigeRecovery::__OnStartSnap() 
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;  

	printf("====================CGigeRecovery::__OnStartSnap()====================\n");

	// Send AcquisitionStart command
	printf("<Send AcquisitionStart command to device>\n");
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_START);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return false;
	}
	m_bIsSanp = true;

	return true;
}

//---------------------------------------------------------------------------------
/**
\brief    This function demonstrates how to stop acquisition

\return   true: Success, false: Fail
*/
//----------------------------------------------------------------------------------
bool CGigeRecovery::__OnStopSnap() 
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;  	
	printf("====================CGigeRecovery::__OnStopSnap()====================\n");
	
	// Send AcquisitionStop command 
	printf("<Send AcquisitionStop command to device>\n");
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return false;
	}
	m_bIsSanp = false;
	
	return true;
}

//---------------------------------------------------------------------------------
/**
\brief    This function demonstrates how to close device.

\return   true: Success, false: Fail
*/
//----------------------------------------------------------------------------------
bool CGigeRecovery::__OnCloseDevice() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	printf("====================CGigeRecovery::__OnCloseDevice()====================\n");

	// Unregister the device OffLine callback function 
	printf("<Unregister device OffLine callback>\n");
	emStatus = GXUnregisterDeviceOfflineCallback(m_hDevice, m_hCB);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
	}
	
	printf("<Close device>\n");
	emStatus = GXCloseDevice(m_hDevice);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
	}
	m_hDevice = NULL;

	// Release memory 
	if (m_stFrameData.pImgBuf != NULL)
	{
		delete[]m_stFrameData.pImgBuf;
		m_stFrameData.pImgBuf = NULL;
	}

	m_bIsOpen = false;

	return true;
}

//---------------------------------------------------------------------------------
/**
\brief    This function demonstrates how to acquire image continuously

\return   void
*/
//----------------------------------------------------------------------------------
void CGigeRecovery::__Acquisition() 
{ 
	GX_STATUS emStatus = GX_STATUS_SUCCESS; 
	printf("====================CGigeRecovery::__Acquisition()====================\n");
	printf("<Press any key to stop acquisition>\n");
	
	
	while(!_kbhit())
	{
		if (m_bIsOffLine)         // The device is OffLine
		{
	       
		   __ProcessOffline();
		   
		
		   __Recovery();
		}
		else                    // The device is OnLine 
		{
			emStatus = GXGetImage(m_hDevice, &m_stFrameData, 500);
			if (emStatus == GX_STATUS_SUCCESS)
			{
				printf("Successfully get Image\n");
			}
			else
			{
				__GetErrorString(emStatus);
			}
		}
	}

	_getch();
}

//---------------------------------------------------------------------------------
/**
\brief    This function demonstrates how to deal with the OffLine event

\return   void
*/
//----------------------------------------------------------------------------------
void CGigeRecovery::__ProcessOffline()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	printf("**********************Process Offline**********************\r");

	// If the device is acquiring image then stop it.
	if (m_bIsSanp)
	{
        // Send AcquisitionStop command  
		printf("\n<Send AcquisitionStop command to device>\n");
		emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			__GetErrorString(emStatus);
		}
		m_bIsSanp = false;
	}

	// If the device is opened then close it.
	if (m_bIsOpen)
	{
		// Unregister the OffLine callback function
	    printf("<Unregister device OffLine callback>\n");
  	    emStatus = GXUnregisterDeviceOfflineCallback(m_hDevice, m_hCB);
	    if (emStatus != GX_STATUS_SUCCESS)
	    {
	       __GetErrorString(emStatus);
	    }
		
		printf("<Close device>\n");
		emStatus = GXCloseDevice(m_hDevice);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			__GetErrorString(emStatus);
		}
		m_hDevice = NULL;
		
		// Release memory
		if (m_stFrameData.pImgBuf != NULL)
		{
			delete[] m_stFrameData.pImgBuf;
			m_stFrameData.pImgBuf = NULL;
		}
		m_bIsOpen = false;
	}
}

//---------------------------------------------------------------------------------
/**
\brief    This function demonstrates how to recovery the device from OffLine status

\return   void
*/
//----------------------------------------------------------------------------------
void CGigeRecovery::__Recovery()
{
	GX_STATUS         emStatus     = GX_STATUS_SUCCESS;   
	uint32_t          nDeviceNum   = 0;    
	int64_t           nPayLoadSize = 0;     
	GX_OPEN_PARAM     stOpenParam;        

	printf("**********************Recovery**********************\r");
	
	emStatus = GXUpdateDeviceList(&nDeviceNum, 1000);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		return;
	}

	// Check if found any device 
	if (nDeviceNum <= 0)
	{
		return;
	}

	// If the device is opened then close it to ensure the camera could open again.
	if (m_hDevice != NULL)
	{
		emStatus = GXCloseDevice(m_hDevice);
		m_hDevice = NULL;
	}
	
	stOpenParam.accessMode = GX_ACCESS_EXCLUSIVE;
	stOpenParam.openMode   = GX_OPEN_MAC;
	stOpenParam.pszContent = m_chMACAddress;
	printf("\n<Open Device by MAC %s>\n", m_chMACAddress);
	emStatus = GXOpenDevice(&stOpenParam, &m_hDevice);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return;
	}

	printf("<Import config file>\n");
	emStatus = GXImportConfigFile(m_hDevice, m_strSavePath.c_str());
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return;
	}
	
	// Allocate memory for acquiring image
	if (m_stFrameData.pImgBuf != NULL)
	{
		delete[] m_stFrameData.pImgBuf;
		m_stFrameData.pImgBuf = NULL;
	}
    emStatus = GXGetInt(m_hDevice, GX_INT_PAYLOAD_SIZE, &nPayLoadSize);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return;
	}
	m_stFrameData.pImgBuf = new BYTE[(size_t)nPayLoadSize];
	if (m_stFrameData.pImgBuf == NULL)
	{
		printf("<Failed to allocate memory>\n");
		return;
	}

	// Register the OffLine callback function
	printf("<Register device OffLine callback>\n");
	emStatus = GXRegisterDeviceOfflineCallback(m_hDevice, this, __OnDeviceOfflineCallbackFun, &m_hCB);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return;
	}
	 m_bIsOpen = true;

	 // Send AcquisitionStart command
	printf("<Send AcquisitionStart command to device>\n");
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_START);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		__GetErrorString(emStatus);
		return;
	}
	m_bIsSanp = true;
	m_bIsOffLine = false;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to convert the error code to error description
\param  emErrorStatus  
 
\return void
*/
//----------------------------------------------------------------------------------
void CGigeRecovery::__GetErrorString(GX_STATUS emErrorStatus)
{
	char      *pchErrorInfo = NULL;
	size_t    nSize         = 0;
	GX_STATUS emStatus      = GX_STATUS_SUCCESS;
	
	// Get the length of error information and allocate memory.
	emStatus = GXGetLastError(&emErrorStatus, NULL, &nSize);
	pchErrorInfo = new char[nSize];
	if (pchErrorInfo == NULL)
	{
		printf("<Failed to allocate memory>\n");
		return ;
	}
	
	// Get error information and show them to user 
	emStatus = GXGetLastError(&emErrorStatus, pchErrorInfo, &nSize);
	if (emStatus != GX_STATUS_SUCCESS)
	{
        printf("<Fail to call GXGetLastError interface!>\n");
	}

	printf("%s\n", (LPCTSTR)pchErrorInfo);

	// Release memory 
	if (pchErrorInfo != NULL)
	{
		delete[]pchErrorInfo;
		pchErrorInfo = NULL;
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to recovery from offLine status     

\return void
*/
//----------------------------------------------------------------------------------
void CGigeRecovery::GigeRecovery()
{
	bool bReturn = false;

	bReturn = __OnOpenDevice();
	if (!bReturn)
	{
		return;
	}

	bReturn = __OnStartSnap();
	if (!bReturn)
	{
		__OnCloseDevice();
		return;
	}

	__Acquisition();	

	__OnStopSnap();
		
	__OnCloseDevice();
}





