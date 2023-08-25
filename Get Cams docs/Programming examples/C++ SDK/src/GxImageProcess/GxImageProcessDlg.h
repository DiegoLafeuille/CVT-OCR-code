// GxImageProcessDlg.h : header file

#pragma once
#include "afxcmn.h"
#include "GalaxyIncludes.h"
#include "GXBitmap.h"

#define PARAM_TRANSFORMATION_VALUE  10.0 ///< The slider control in the UI accepts only integer values,but Gamma value and Sharpen value are decimal,
                                         ///< So we need get the device Gamma and Sharpen value and convert its to an integer value

// CGxImageProcessDlg dialog
class CGxImageProcessDlg : public CDialog
{
public:
	//---------------------------------------------------------------------------------
	/**
	\brief   This is a acquisition callback class.
	*/
	//----------------------------------------------------------------------------------
	class CSampleCaptureEventHandler : public ICaptureEventHandler
	{
	public:
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
				CGxImageProcessDlg*  pImgProDlg = (CGxImageProcessDlg*)pUserParam;
				pImgProDlg->ShowImageProcess(objImageDataPointer);
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
	explicit CGxImageProcessDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	enum { IDD = IDD_GXIMAGEPROCESS_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
	

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnClose();
	afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
	afx_msg void OnBnClickedOpenDevice();
	afx_msg void OnBnClickedCloseDevice();
	afx_msg void OnBnClickedStartSnap();
	afx_msg void OnBnClickedStopSnap();
	afx_msg void OnCbnSelchangeComboAwb();
	afx_msg void OnCbnSelchangeComboAwbLight();
	afx_msg void OnBnClickedAidcOpenAccelerateccelerateCheck();
	afx_msg void OnBnClickedOpenBadPixelCorrect();
	afx_msg void OnBnClickedOpenDenoise();
	afx_msg void OnBnClickedOpenColorCorrect();
	afx_msg void OnBnClickedOpenSharpen();
	afx_msg void OnCbnDropdownComboAwb();
	afx_msg void OnCbnDropdownComboAwbLight();
	DECLARE_MESSAGE_MAP()

public:
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	afx_msg void OnTimer(UINT_PTR nIDEvent);


	/// Image processing function
	void ShowImageProcess(CImageDataPointer& objImageDataPointer);

private:

	/// initialize UI
	void __InitUI(); 

	/// Initializes the enumeration type UI
	int __InitEnumUI(const gxstring& strFeaturName, CComboBox *pComboBox, bool bIsImplemented);

	/// initialize device param
	void __InitParam(); 

	/// Update UI
	void __UpdateUI();

	/// Change the width of the ComboBox drop-down box
	void __ChangeComBoxWidth(CComboBox* pComBox);

 	/// Determine whether the camera turns on color correction
    void __IsEnableDeviceColorCorrection();

    /// Determine if the camera is enabled by Gamma enabling
    void __IsEnableDeviceGamma();

    /// Determine whether the camera turns on sharpening enable
    void __IsEnableDeviceSharpen();

	/// Initializes the drop-down box
	void __InitCombo();

	HICON                              m_hIcon;
	CGXDevicePointer                   m_objDevicePtr;                     ///< The handle for device 
	CGXStreamPointer                   m_objStreamPtr;                     ///< The object for stream  
	CGXFeatureControlPointer           m_objFeatureControlPtr;             ///< The object for device feature control 
	CGXFeatureControlPointer           m_objStreamFeatureControlPtr;       ///< The object for stream feature control
	CImageProcessConfigPointer         m_objImageProcessPtr;               ///< The object for image process
	CGXBitmap*                         m_pBitmap;                          ///< The object for display and drawing
	CWnd*                              m_pWnd;                             ///< The handle for window 
	CSampleCaptureEventHandler*        m_pSampleCaptureEventHandler;       ///< The pointer for callback function
	bool                               m_bIsOpen;                          ///< The flag indicates whether the device has been opened
	bool                               m_bIsSnap;                          ///< The flag indicates whether image acquisition has started
	bool                               m_bIsWhiteBalance;                  ///< The flag indicates whether the device supports WhiteBalance
	bool                               m_bIsWhiteBalanceLight;             ///< The flag indicates whether the device supports WhiteBalanceLight
	int                                m_nComBoAWBOldValue;                ///< BalanceWhiteAuto value
	int                                m_nComBoAWBLightOldValue;           ///< BalanceWhiteAuto light value
	int                                m_nOldSliderSharpenValue;           ///< Record sharpen value
	int                                m_nOldSliderGammaValue;             ///< Record Gamma value
	int                                m_nOldSliderContrastValue;          ///< Record contrast value
	int                                m_nOldSliderLightnessValue;         ///< Record lightness value
	int                                m_nOldSliderSaturationValue;        ///< Record saturation value
	std::string                        m_strBalanceWhiteAutoValue;         ///< Record BalanceWhiteAuto mode
	bool                               m_bEnableColorCorrect;              ///< Color correction enable switch
    bool                               m_bEnableGamma;                     ///< Gamma enable switch
    bool                               m_bEnableSharpness;                 ///< Sharpen enable switch
	bool                               m_bEnableStaturation;               ///< Saturation enable switch
	bool                               m_bEnableCheckDenoise;              ///< Denoise reduction enable switch
	bool                               m_bIsColorFilter;                   ///< Open the camera for the color camera flag
	bool                               m_bEnumDevices;                     ///< Whether the device is enumerated to the device flag
	CSliderCtrl                        m_sliderSharpen;
	CSliderCtrl                        m_sliderGamma;
	CSliderCtrl                        m_sliderContrast;
	CSliderCtrl                        m_sliderLightness;
	CSliderCtrl                        m_sliderSaturation;
	BOOL                               m_bCheckAcceleract;
	BOOL                               m_bCheckPixelCorrect;
	BOOL                               m_bCheckDenoise;
	BOOL                               m_bCheckColorCorrect;
	BOOL                               m_bCheckSharpen;
	CComboBox                          m_comboChooseDevice;
	GxIAPICPP::gxdeviceinfo_vector     m_vecDeviceInfo;
};
