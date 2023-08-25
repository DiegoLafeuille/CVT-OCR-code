//------------------------------------------------------------
/**
\file    DeviceProcess.h 
\brief   Device Process 
*/
//------------------------------------------------------------
#pragma once
#include "GalaxyIncludes.h"
#include "GXBitmap.h"
#include "Fps.h"

//---------------------------------------------------------------------------------
/**
\brief   Device process class
*/
//----------------------------------------------------------------------------------
class CDeviceProcess
{
public:

	CDeviceProcess(void);
	~CDeviceProcess(void);

	//---------------------------------------------------------------------------------
	/**
	\brief   This is a acquisition callback class.
	*/
	//----------------------------------------------------------------------------------
	class CSampleCaptureEventHandler :public ICaptureEventHandler
	{
		//---------------------------------------------------------------------------------
		/**
		\brief   This is a acquisition callback function that demonstrates how to record the interval that from sending a TriggerSoftware command to receiving a image.
		\param    objImageDataPointer		image information
		\param	  pUserParam                user private param

		\return  void
		*/
		//----------------------------------------------------------------------------------
		void DoOnImageCaptured(CImageDataPointer& objImageDataPointer, void* pUserParam)
		{
			try
			{
				CDeviceProcess* pDeviceProcess = (CDeviceProcess*)pUserParam;

				//Show Image
				pDeviceProcess->__ShowPicture(objImageDataPointer);
			}
			catch (CGalaxyException)
			{
				//do nothing

			}
			catch (std::exception)
			{
				//do nothing
			}
		}
	};

public:
	/// Open device
	void OpenDevice(gxstring strDeviceSN, CWnd* m_pWnd,int nDeviceIndex);

	/// Close device
	void CloseDevice();

	/// start acquisition
	void StartSnap();

	/// stop acquisition
	void StopSnap();

	/// Get the flag of device opening
	bool IsOpen() const;

	/// Get the flag of starting acquisition
	bool IsSnap() const;

	/// Set whether the frame rate flag is displayed
	void SetShowFPS(bool bIsShowFps);

	/// Get whether the frame rate flag is displayed
	bool GetShowFPS();

	/// Refresh the device model and frame rate information
	void RefreshDeviceSNFPS();

	CGXFeatureControlPointer    m_objFeatureControlPtr;           ///< The handle for feature control
	CGXFeatureControlPointer    m_objStreamFeatureControlPtr;     ///< The object for stream feature control

private:

	/// Show Image
	void __ShowPicture(CImageDataPointer& objImageDataPointer);

	CGXDevicePointer            m_objDevicePtr;                  ///< The handle for device
	CGXStreamPointer            m_objStreamPtr;                  ///< The handle for device stream
	CGXBitmap*                  m_pBitmap;                       ///< The pointer for drawing
	CSampleCaptureEventHandler* m_pSampleCaptureEventHandle;     ///< The pointer for callback
	bool                        m_bIsOpen;                       ///< The flag indicates whether the device has been opened
	bool                        m_bIsSnap;                       ///< The flag indicates whether image acquisition has started
	bool                        m_bShowDeviceFPS;                ///< The flag indicates whether the frame rate has displayed
	char                        *m_pRecordDeviceSNFPS;           ///< Device model and frame rate information
	gxstring                    m_strDeviceSN;                   ///< Device serial number
	CFps                        m_objfps;                        ///< The frame rate
	int							m_nDeviceIndex;                  ///< Device Index

};
