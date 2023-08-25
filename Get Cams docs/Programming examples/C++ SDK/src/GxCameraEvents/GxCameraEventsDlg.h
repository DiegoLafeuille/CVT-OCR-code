//----------------------------------------------------------------------------------
/**
\brief  CGxCameraEventsDlg class declaration
*/
//----------------------------------------------------------------------------------
#pragma once
#include "GalaxyIncludes.h"
#include "StopWatch.h"

// CGxCameraEventsDlg dialog
class CGxCameraEventsDlg : public CDialog
{
public:

	/// Construction
	CGxCameraEventsDlg(CWnd* pParent = NULL);	

	/// Dialog Data
	enum { IDD = IDD_GXCAMERAEVENTS_DIALOG };

protected:

	/// DDX/DDV support
	virtual void DoDataExchange(CDataExchange* pDX);	

protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:

	afx_msg void OnBnClickedOpenDevice();
	afx_msg void OnBnClickedStartSnap();
	afx_msg void OnBnClickedStopSnap();
	afx_msg void OnBnClickedCloseDevice();
	afx_msg void OnBnClickedSendSoftCommand();
	afx_msg void OnClose();

public:

	// print time information to windows
	void PrintInfotoWnd(std::string &strInfo);
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	CStopWatch                      m_objImageTime;               ///< The stopwatch for receiving image
	CStopWatch                      m_objShutterTime;             ///< The stopwatch for completing the exposure event
	bool                            m_bIsTrigValid;               ///< Valid flag for trigger: sets this flag to false when a trigger is executing

	//---------------------------------------------------------------------------------
	/**
	\brief    This is a acquisition callback class.
	*/
	//----------------------------------------------------------------------------------
	class CSampleCaptureEventHandle :public ICaptureEventHandler
	{
	public:

		//---------------------------------------------------------------------------------
		/**
		\brief   This is a acquisition callback function that demonstrates how to record the interval that from sending a TriggerSoftware command to receiving a image.
		\param    objImageDataPointer		image information
		\param	  pUserParam                user private param

		\return   void
		*/
		//----------------------------------------------------------------------------------
		void DoOnImageCaptured(CImageDataPointer& objImageDataPointer, void* pUserParam)
		{
			CGxCameraEventsDlg* pDlg      = (CGxCameraEventsDlg*)pUserParam;

			try
			{
				double              dSnaptime = 0;
				char                chTemp[8] = "ms";

				//Get the time for acqusition a frame
				dSnaptime = pDlg->m_objImageTime.Stop();
				CString strInformation = "";
				strInformation.Format("The time interval that from sending a TriggerSoftware command to receiving a image: %0.2f%s",dSnaptime,chTemp);
				std::string strPrintInformation = strInformation.GetBuffer(0);
				pDlg->PrintInfotoWnd(strPrintInformation);		
			}
			catch (CGalaxyException)
			{
				//do nothing
			}
			catch (std::exception)
			{
				//do nothing
			}

			pDlg->m_bIsTrigValid = true;
		}
	};

	//---------------------------------------------------------------------------------
	/**
	\brief    This is a event callback class
	*/
	//----------------------------------------------------------------------------------
	class CSampleFeatureEventHandle :public IFeatureEventHandler
	{
	public:

		//---------------------------------------------------------------------------------
		/**
		\brief    This is a event callback function that demonstrates how to process the ExposureEndEvent and record the interval that from sending a TriggerSoftware command to receiving a ExposureEndEvent
		\param    strFeatureName		Event name
		\param	  pUserParam            user private param

		\return   void
		*/
		//----------------------------------------------------------------------------------
		void DoOnFeatureEvent(const GxIAPICPP::gxstring& strFeatureName, void* pUserParam)
		{
			try
			{
				CGxCameraEventsDlg*  pDlg             = (CGxCameraEventsDlg*)pUserParam;
				double              dElapsedtime      = 0;
				char                chTemp[8]         = "ms";
				if (strFeatureName == "EventExposureEnd")
				{
					// stop timing
					dElapsedtime = pDlg->m_objShutterTime.Stop();

					// record exposure end timing 
					CString strInformation = "";
					strInformation.Format("The time interval that from sending a TriggerSoftware command to receiving a ExposureEndEvent: %0.2f%s",dElapsedtime,chTemp);
					std::string strPrintInformation = strInformation.GetBuffer(0);
					pDlg->PrintInfotoWnd(strPrintInformation);
				}
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

private:

	// Check whether the device supports ExposureEndEvent
	bool __IsSupportExposureEnd();

	// initialize device param
	void __InitParam();

	// Update UI
	void __UpdateUI();

	CGXDevicePointer                m_objDevicePtr;               ///< The handle for device 
	CGXStreamPointer                m_objStreamPtr;               ///< The object for stream
	CGXFeatureControlPointer        m_objFeatureControlPtr;       ///< The object for device feature control
	CGXFeatureControlPointer        m_objStreamFeatureControlPtr; ///< The object for stream feature control
	GX_FEATURE_CALLBACK_HANDLE      m_hCB;                        ///< The handle for ExposureEndEvent callback function
	void*                           m_pLogWnd;                    ///< The pointer for recording the time of the event
	CSampleCaptureEventHandle*      m_pCaptureEventHandle;        ///< The pointer for capture callback
	CSampleFeatureEventHandle*      m_pFeatureEventHandle;        ///< The pointer for ExposureEndEvent callback
	bool                            m_bIsOpen;                    ///< The flag indicates whether the first device has been opened
	bool                            m_bIsSnap;                    ///< The flag indicates whether AcquisitionStart command has been send 
};