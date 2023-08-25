//----------------------------------------------------------------------------------
/**
\file    GxGigeRecovery.h 
\brief   CGxGigeRecovery class declaration
*/
//----------------------------------------------------------------------------------
#pragma once
#include "GalaxyIncludes.h"
#include "conio.h"
#include <iostream>


class CSampleDeviceOfflineEventHandle;              ///< Declare device offline event class

//--------------------------------------------------
/**
\brief     Device control class
*/
//--------------------------------------------------
class CGigeRecovery
{
public:

	/// Construction
	CGigeRecovery();
 
	/// Destruction
	virtual ~CGigeRecovery();

	/// The external interface function for OffLine recovery
	void GigeRecovery();

	bool m_bIsOffline;               ///< The flag indicates whether the device is offline

private:

	/// Open device
	bool __OnOpenDevice() throw();

	/// start acquisition
	bool __OnStartSnap() throw();

	/// stop acquisition
	bool __OnStopSnap() throw();

	/// Close device
	bool __OnCloseDevice() throw();

	/// Initialize the device parameters
	void __InitParam();

	/// continuous acquisition
	void __Acquisition() throw();

	/// Deal with the offline event
	void __ProcessOffline() throw();

	/// Recovery the device from OffLine status
	void __Recovery() throw();

	CGXDevicePointer                   m_objDevicePtr;                       ///< The handle for device
	CGXStreamPointer                   m_objStreamPtr;                       ///< The handle for stream
	CGXFeatureControlPointer           m_objFeatureControlPtr;               ///< The handle for feature control
	CGXFeatureControlPointer           m_objStreamFeatureControlPtr;         ///< The object for stream feature control
	GX_DEVICE_OFFLINE_CALLBACK_HANDLE  m_hCB;                                ///< The handle for OffLine callback event
	CImageDataPointer                  m_objImageDataPtr;                    ///< The pointer for acquired image data 

	bool                               m_bIsOpen;                            ///< The flag indicates whether the first device has been opened
	bool                               m_bIsSnap;                            ///< The flag indicates whether AcquisitionStart command has been send   
	GxIAPICPP::gxstring                m_strSavePath;                        ///< The path for saving device configuration parameters
	CSampleDeviceOfflineEventHandle*   m_pDeviceOfflineEventHandle;          ///< The pointer for OffLine callback event
	gxstring                           m_strMACAddress;                      ///< The MAC address for device
};

//--------------------------------------------------
/**
\brief     This is a offline callback class.
*/
//--------------------------------------------------
class CSampleDeviceOfflineEventHandle :public IDeviceOfflineEventHandler
{
public:
	//--------------------------------------------------
	/**
	\brief    This is a offline callback function that demonstrates how to record the device is offline
    \param     pUserParam      user private param
    
	\return    void
	*/
	//--------------------------------------------------
	void DoOnDeviceOfflineEvent(void* pUserParam)
	{
		try
		{
			CGigeRecovery* pGigeRecovery = (CGigeRecovery*)pUserParam;
			pGigeRecovery->m_bIsOffline = true;
			std::cout<<"**********************Device offline**********************"<<std::endl;

		}
		catch (CGalaxyException)
		{
			//do nothing
		}
		catch (std::exception)
		{
			//do noting
		}
	}
};