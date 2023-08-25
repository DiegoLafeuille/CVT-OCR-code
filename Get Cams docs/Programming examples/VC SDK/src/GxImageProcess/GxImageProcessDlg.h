// GxImageProcessDlg.h : header file
//

#if !defined(AFX_GXIMAGEPROCESSDLG_H__FA4E5FCA_79E2_40DF_9164_D9FDEE73A8DB__INCLUDED_)
#define AFX_GXIMAGEPROCESSDLG_H__FA4E5FCA_79E2_40DF_9164_D9FDEE73A8DB__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

/////////////////////////////////////////////////////////////////////////////
// CGxImageProcessDlg dialog
#include "GxIAPI.h"
#include "DxImageProc.h"
#include "StdAfx.h"
#include <vector>
#include <stdio.h>
using namespace std;

#define   CALCCCPARAM_LENGTH               (sizeof(VxInt16) * 9)      ///< Color control array length
#define   DEFAULT_LIGHTNESS                0                          ///< Default brightness value
#define   DEFAULT_SHARPEN                  0.1f                       ///< Default sharpening value
#define   DEFAULT_SATURATION               64                         ///< Default saturation value
#define   DEFAULT_CONTRAST                 0                          ///< Default contrast value
#define   DEFAULT_GAMMA                    1                          ///< The default Gamma value
#define   DEFAULT_COLOR_CORRECTION         0                          ///< Default color correction
#define   WHITE_BALANCE_RNTRY              3                          ///< Automatic white balanced enumerator


///< Error: function macro definition
#define GX_VERIFY(emStatus) \
	                        if (emStatus != GX_STATUS_SUCCESS)\
							{\
						       ShowErrorString(emStatus); \
							   return ;\
							} 
///< Determine the return value macro definition
#define VERIFY_STATUS_RET(emStatus) \
	                               if (emStatus != GX_STATUS_SUCCESS) \
								   {\
								      return emStatus;\
								   }\

///< Determine the return value macro definition
#define VERIFY_STATUS_BREAK(emStatus) \
    if(emStatus != GX_STATUS_SUCCESS) \
    {\
        break;\
    }

class CGxImageProcessDlg : public CDialog
{
// Construction
public:
	explicit CGxImageProcessDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	//{{AFX_DATA(CGxImageProcessDlg)
	enum { IDD = IDD_GXIMAGEPROCESS_DIALOG };
	CSliderCtrl	m_sliderGamma;
	CSliderCtrl	m_sliderContrast;
	CSliderCtrl	m_sliderLightness;
	CSliderCtrl	m_sliderSaturation;
	CSliderCtrl	m_sliderSharpen;
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CGxImageProcessDlg)
	virtual BOOL PreTranslateMessage(MSG* pMsg);
protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL



	// Generated message map functions
	//{{AFX_MSG(CGxImageProcessDlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnOpenDevice();
	afx_msg void OnStartSnap();
	afx_msg void OnStopSnap();
	afx_msg void OnCloseDevice();
	afx_msg void OnSelchangeComboAwb();
	afx_msg void OnSelchangeComboAwbLight();
	afx_msg void OnClose();
	afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
	afx_msg void OnOpenBadPixelCorrect();
	afx_msg void OnOpenDenoise();
	afx_msg void OnOpenAccelerate();
	afx_msg void OnOpenSharpen();
	afx_msg void OnOpenColorCorrect();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
public:
	/// initialization parameter 
	GX_STATUS InitParam();
	
	/// The data format is set to 8bit
	GX_STATUS SetPixelFormat8bit();

	/// Error message prompt
	void ShowErrorString(GX_STATUS emErrorStatus);

	/// Prepare resources for image display
	bool PrepareForShowImage();

	/// Release resources to prepare for image display
	void UnPrepareForShowImage();

	/// Camera callback function
	static void __stdcall OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM* pFrame);

	/// display image
	void DrawImage(BYTE *pImageBuf, int nWidth, int nHeight, BITMAPINFO *pBmpInfo);

	/// Initializes the enumeration type UI interface
	void InitEnumUI(GX_FEATURE_ID_CMD emFeatureID, CComboBox *pComboBox, bool bIsImplement);

	/// Image correlation parameter
	GX_STATUS GetImageParam();

	/// Initializes the image processing functional structure
	bool InitImageProcessStruct();

	/// Release the resources requested by the image processing function
	void UnInitImageProcessStruct();
	
	/// Initialize the UI interface
	void InitUI();

	/// Update UI interface
	void UpdateUI();

private:
    
    /// Gets whether the device opens color correction
    GX_STATUS __IsEnableDeviceColorCorrect();

    /// Gets whether the device opens the Gamma enabled
    GX_STATUS __IsEnableDeviceGamma();

    /// Gets whether the device opens sharpening enable
    GX_STATUS __IsEnableDeviceSharpness();

	/// Initial drop-down list control
	void __InitCombo();

	/// Initializes some parameters of the color camera
	GX_STATUS __InitColorParam();

	/// Initializes the color image processing structure
	bool __InitColorStruct();

	/// Initializes the black and white image processing structure
	bool __InitMonoStruct();

	/// Mono image processing algorithm in callback function
	void OnMonoImageProcess();

	/// Color image processing algorithm in callback function
	void OnColorImageProcess();


	GX_DEV_HANDLE       m_hDevice;              ///< The handle for device 
	BITMAPINFO          *m_pBmpInfo;	        ///< The pointer for showing image
	BYTE                *m_pBufferRaw;          ///< The pointer for the raw data of image
	BYTE                *m_pBufferImage;	    ///< RGB/Mono Image data for display
	BYTE                *m_pBufferMono;         ///< Mono Image data for black and white image processing after completion of the rollover
    char                m_chBmpBuf[2048];	    ///< The buffer for BMP image
	int64_t             m_nPayLoadSize;         ///< PayLoadSize  
	int64_t             m_nImageHeight;         ///< ImageHeight
    int64_t             m_nImageWidth;          ///< ImageWidth
	int64_t             m_nPixelColorFilter;    ///< PixelColorFilter
	bool                m_bColorFilter;         ///< The flag indicates whether the device supports PixelColorFilter 
	bool                m_bAwbLampHouse;        ///< The flag indicates whether the device supports AwbLampHouse 
	bool                m_bColorCorrection;     ///< The flag indicates whether the device supports ColorCorrection
	bool                m_bIsOpenCC;            ///< The flag indicates whether the ColorCorrection is opened
	bool                m_bIsOpen;              ///< The flag indicates whether the first device has been opened
	bool                m_bIsSnap;              ///< The flag indicates whether image acquisition has started
	int64_t             m_nGetColorCorrection;  ///< The acquired value of ColorCorrection 
	int64_t             m_nColorCorrection;     ///< The value of ColorCorrection 
	int64_t             m_nContrast;            ///< The value of Contrast 
	int64_t             m_nAutoWhiteBalance;    ///< The value of AutoWhiteBalance
	double              m_dGamma;               ///< The value of Gamma 
	int                 m_nLightness;           ///< The value of Lightness 
	int                 m_nSaturation;          ///< The value of Saturation 
	int                 m_nPreSaturation;       ///< The value of Saturation before operation 
	int64_t             m_nPreContrast;         ///< The value of Contrast before operation
	int64_t             m_nPreColorCorrection;  ///< The value of ColorCorrection before operation
	int                 m_nPreLightness;        ///< The value of Lightness before operation
	double              m_dPreGamma;            ///< The value of Gamma before operation
	float               m_fSharpen;             ///< The value of Sharpen 
	COLOR_IMG_PROCESS   m_objImageProcess;      ///< The structure of image processing 
	bool                m_bUpData;              ///< The flag indicates whether the showing CPU information is updated 
	HWND                m_hWndHandle;           ///< The handle for showing image 
	HDC                 m_hDC;                  ///< The handle for drawing image DC 
	RECT                m_objRect;              ///< The size of image window
	bool                m_bEnableColorCorrect;  ///< Color correction enables flag bit
    bool                m_bEnableGamma;         ///< Gamma enabled flag bit
    bool                m_bEnableSharpness;     ///< Sharpen enable flag bit
	bool                m_bEnableOpen;          ///< Open device key enable flag bit
	MONO_IMG_PROCESS    m_objMonoImgProc;       ///< Black and white image processing function sets the structure
	bool                m_bSharpness;           ///< Sharpen
	bool                m_bAccelerate;          ///< Algorithm to accelerate
	bool                m_bDenoise;				///< Identifies whether the algorithm acceleration check box is selected
	bool                m_bDefectivePixelCorrect; ///< Bad point correction
	bool                m_bImageProcess;        ///< Identifies whether the program is performing image processing
	vector<GX_DEVICE_BASE_INFO>   m_vecDevInfo; ///< An array of device basic information structures
	GX_DEVICE_BASE_INFO *m_pBaseinfo;           ///< Device basic information structure pointer
	// Implementation
	HICON m_hIcon;
public:
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	CComboBox m_comboChooseDevice;
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_GXIMAGEPROCESSDLG_H__FA4E5FCA_79E2_40DF_9164_D9FDEE73A8DB__INCLUDED_)
