// GxSingleCamDlg.h : header file
//
#pragma once

#include "GalaxyIncludes.h"
#include "GXBitmap.h"


// CGxSingleCamDlg Dialog
class CGxSingleCamDlg : public CDialog
{
	//---------------------------------------------------------------------------------
	/**
	\brief   inherit the acquisition event handling class
	*/
	//----------------------------------------------------------------------------------
	class CSampleCaptureEventHandler :public ICaptureEventHandler
	{
		//---------------------------------------------------------------------------------
		/**
		\brief   The capture callback function
		\param   objImageDataPointer      image process parameter
		\param   pFrame                   custom defined parameter

		\return  void
		*/
		//----------------------------------------------------------------------------------
		void DoOnImageCaptured(CImageDataPointer& objImageDataPointer, void* pUserParam)
		{
			try
			{
				CGxSingleCamDlg* pSingleCamDlg = (CGxSingleCamDlg*)pUserParam;

				//Display image
				pSingleCamDlg->m_pBitmap->Show(objImageDataPointer);

				//Check whether is need to save image or not.
				if (pSingleCamDlg->m_bCheckSaveBmp == TRUE)
				{
					pSingleCamDlg->SavePicture(objImageDataPointer);
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
// Construction
public:
	CGxSingleCamDlg(CWnd* pParent = NULL);	// standard Construction

// Dialog Data
	enum { IDD = IDD_GXSINGLECAM_DIALOG };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

public:
	afx_msg void OnBnClickedBtnOpenDevice();
	afx_msg void OnBnClickedBtnCloseDevice();
	afx_msg void OnBnClickedBtnStartSnap();
	afx_msg void OnBnClickedBtnStopSnap();
	afx_msg void OnCbnSelchangeComboTriggerMode();
	afx_msg void OnCbnSelchangeComboTriggerSource();
	afx_msg void OnCbnSelchangeComboTriggerActive();
	afx_msg void OnBnClickedCheckSaveBmp();

public:

	virtual BOOL PreTranslateMessage(MSG* pMsg);
	afx_msg void OnClose();
	afx_msg void OnCbnSelchangeComboBalanceWhiteAuto();
	afx_msg void OnCbnSelchangeComboBalanceWhiteSelect();
	afx_msg void OnBnClickedBtnSofttrigger();
	afx_msg void OnEnKillfocusEditShutter();
	afx_msg void OnEnKillfocusEditGain();
	afx_msg void OnEnKillfocusEditBalanceWhiteSerise();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	double m_dEditShutterValue;
	double m_dEditGainValue;
	double m_dEditBalanceRatioValue;
	BOOL   m_bCheckSaveBmp;

public:

	/// save image as bmp format
	void SavePicture(CImageDataPointer& objImageDataPointer);

	CGXBitmap*  m_pBitmap;                  ///< the pointer of saving image class

private:

	///Iinitialize device param
	void __InitParam();

	/// Initialize combox
	int __InitComBox(gxstring strName, CComboBox* pComboBox, bool bIsImplemented);

	/// Initialize ExposureTime Widget
	void __InitShutterTime();

	/// Initialize Gain Widget
	void __InitGain();

	/// Initialize white balance ratio Widget
	void __InitBalanceWhiteRatio();

	/// Initialize UI
	void __InitUI();

	/// Update UI
	void __UpdateUI();	

	/// Set the value of ComboBox for device
	void __SetEnumToDevice(gxstring strEnumName,CComboBox* pCombo);

	CGXDevicePointer                  m_objDevicePtr;             ///< The device handle
	CGXStreamPointer                  m_objStreamPtr;             ///< The stream handle
	CGXFeatureControlPointer          m_objFeatureControlPtr;     ///< The object for device feature control
	CGXFeatureControlPointer          m_objStreamFeatureControlPtr; ///< The object for stream feature control
	CWnd*                             m_pWnd;                     ///< The pointer of Window
	CSampleCaptureEventHandler*       m_pSampleCaptureEventHandle;///< The pointer of capture callback function

	bool                              m_bIsOpen;                  ///< The flag indicates whether the device has been opened.
	bool                              m_bIsSnap;                  ///< The flag indicates whether the device has been started acquisition.
	bool                              m_bColorFilter;             ///< The flag indicates whether the device is supported color filter or not.
	bool                              m_bTriggerMode;             ///< The flag indicates whether the device is supported trigger or not.
	bool                              m_bTriggerSource;           ///< The flag indicates whether the device is supported selecting trigger source. or not.
	bool                              m_bTriggerActive;           ///< The flag indicates whether the device is supported changing trigger activation or not.
	bool                              m_bBalanceWhiteAuto;        ///< The flag indicates whether the device has been opened auto white balance or doesn't been.
	bool                              m_bBalanceWhiteRatioSelect; ///< The flag indicates whether the device is supported selecting balance ratio selector.
	double                            m_dShutterValueMax;         ///< The max value of exposure time      
	double                            m_dShutterValueMin;         ///< The min value of exposure time     
	double                            m_dGainValueMax;            ///< The max value of gain
	double                            m_dGainValueMin;            ///< The min value of gain
	double                            m_dBalanceWhiteRatioMax;    ///< The max value of auto white balance ratio
	double                            m_dBalanceWhiteRatioMin;    ///< The min value of auto white balance ratio
	int                               m_nTriggerModeOld;          ///< Record the value of trigger mode
	int                               m_nTriggerSourceOld;        ///< Record the value of trigger source
	int                               m_nTriggerActiveOld;        ///< Record the value of trigger activation
	int                               m_nBalanceWhiteAutoOld;     ///< Record the value of auto white balance
	int                               m_nBanlanceWhiteRatioOld;   ///< Record the value oof auto white balance ratio
	gxstring                          m_strBalanceWhiteAutoMode;  ///< Record the value of auto white balance mode
	std::string                       m_strSavePath;              ///< Record the path for saving images
};
