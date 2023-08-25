// GxIOSampleDlg.h : header file
//

#if !defined(AFX_GXIOSAMPLEDLG_H__26540673_FD9B_4799_889E_E05C3076016C__INCLUDED_)
#define AFX_GXIOSAMPLEDLG_H__26540673_FD9B_4799_889E_E05C3076016C__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

/////////////////////////////////////////////////////////////////////////////
// CGxIOSampleDlg dialog
#include "GxIAPI.h"
#include "DxImageProc.h"
#include <string>
#include <iostream>
using namespace std;

#define OUTPUT_ACTIVATION_FALSE  0
#define OUTPUT_ACTIVATION_TRUE   1

#define LINE_INVERTER_FALSE 0
#define LINE_INVERTER_TRUE  1

#define  GX_VERIFY(emStatus) \
	if (emStatus != GX_STATUS_SUCCESS)\
							{\
							ShowErrorString(emStatus); \
							return ;\
							 } 

#define VERIFY_STATUS_RET(emStatus) \
	if (emStatus != GX_STATUS_SUCCESS) \
								   {\
								   return emStatus;\
								   }\

class CGxIOSampleDlg : public CDialog
{
// Construction
public:
	CGxIOSampleDlg(CWnd* pParent = NULL);	// standard constructor

public:

	static void __stdcall OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM* pFrame);

	GX_STATUS InitDevice();

    GX_STATUS SetPixelFormat8bit();

	GX_STATUS GetDeviceParam();

	void InitUI();

	void InitEnumUI(GX_FEATURE_ID_CMD emFeatureID, CComboBox *pComboBox, bool bIsImplemented);

	void InitGainUI();

	void InitShutterUI();

	bool PrepareForShowColorImg();

	bool PrepareForShowMonoImg();

	bool PrepareForShowImg();

	void UnPrepareForShowImg();

    void InitOutputValue();

    void DrawImg(BYTE *pImageBuffer);

	void UpDateUI();

	void ShowErrorString(GX_STATUS emErrorStatus);

	void InitTriggerModeUI();

	void InitTriggerSourceUI();

	void InitTriggerActivationUI();

	void InitStrobeSwitchUI();

	void InitOutputModeUI();

	void InitOutputSelectorUI();

	void InitLineSelector();

	void InitLineMode();

	void InitLineInverter();

	void InitLineSource();
 
	void UpdateLineRelativeInfo();

	void UpdateUserOutputRelativeInfo();

public:
	GX_DEV_HANDLE           m_hDevice;            ///< The handle for device  
	
	GX_FLOAT_RANGE          m_stShutterFloatRange;///< The range of ExposureTime 
	GX_FLOAT_RANGE          m_stGainFloatRange;   ///< The range of Gain
	
	int64_t                 m_nImageWidth;        ///< ImageWidth
	int64_t                 m_nImageHeight;       ///< ImageHeight
	int64_t                 m_nPayLoadSize;       ///< PayLoadSize
	int64_t                 m_nPixelColorFilter;  ///< PixelColorFilter
	int64_t                 m_nTriggerMode;       ///< TriggerMode
	BITMAPINFO             *m_pBmpInfo;	          ///< The pointer for showing image 

    BOOL                    m_bDevOpened;         ///< The flag indicates whether the first device has been opened
    BOOL                    m_bIsSnap;            ///< The flag indicates whether image acquisition has started
	bool                    m_bIsColorFilter;     ///< The flag indicates whether the device support PixelColorFilter

	BYTE                   *m_pRawBuffer;         ///< The pointer for the raw data of image
	BYTE                   *m_pImgBuffer;         ///< The pointer for the image has been processed 
	char                    m_chBmpBuf[2048];     ///< The buffer for BMP image

	bool                    m_bTriggerMode;       ///< The flag indicates whether the device supports TriggerMode
	bool                    m_bTriggerActive;     ///< The flag indicates whether the device supports TriggerActive
	bool                    m_bTriggerSource;     ///< The flag indicates whether the device supports TriggerSource
	bool                    m_bOutputSelector;    ///< The flag indicates whether the device supports OutputSelector
	bool                    m_bOutputMode;        ///< The flag indicates whether the device supports OutputMode
	bool                    m_bOutputValue;       ///< The flag indicates whether the device supports OutputValue
	bool                    m_bStrobeSwitch;      ///< The flag indicates whether the device supports StrobeSwitch
	
	bool                    m_bLineSelector;      ///< The flag indicates whether the device supports LineSelector;
	bool                    m_bLineMode;          ///< The flag indicates whether the device supports LineMode
	bool                    m_bLineInverter;      ///< The flag indicates whether the device supports LineInverter
	bool                    m_bLineSource;        ///< The flag indicates whether the device supports LineSource
	bool                    m_bStrobeSwitchIsUse; ///< The flag indicates whether the StrobeSwitch is available

	CWnd                   *m_pWnd;               ///< The pointer for showing image window 
	HDC                     m_hDC;                ///< The handle for drawing image DC         

// Dialog Data
	//{{AFX_DATA(CGxIOSampleDlg)
	enum { IDD = IDD_GXIOSAMPLE_DIALOG };
	double	m_dGainValue;
	double	m_dShutterValue;
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CGxIOSampleDlg)
	public:
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CGxIOSampleDlg)
	virtual BOOL OnInitDialog();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg void OnBnClickedBtnOpenDevice();
	afx_msg void OnCbnSelchangeComboOutputSource();
	afx_msg void OnCbnSelchangeComboOutputMode();
	afx_msg void OnCbnSelchangeComboOutputValue();
	afx_msg void OnClose();
	afx_msg void OnCbnSelchangeComboTriggerSource();
	afx_msg void OnCbnSelchangeComboTriggerMode();
	afx_msg void OnBnClickedBtnSendSoftTrigger();
	afx_msg void OnCbnSelchangeComboTriggerActivation();
	afx_msg void OnBnClickedBtnStartSnap();
	afx_msg void OnCbnSelchangeComboStrobeSwitch();
	afx_msg void OnBnClickedBtnStopSnap();
	afx_msg void OnBnClickedBtnCloseDevice();
	afx_msg void OnKillfocusEditShutterValue();
	afx_msg void OnKillfocusEditGainValue();
	afx_msg void OnCbnSelchangeComboLineSelector();
	afx_msg void OnCbnSelchangeComboLineMode();
	afx_msg void OnCbnSelchangeComboLineInverter();
	afx_msg void OnCbnSelchangeComboLineSource();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
	
};
//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_GXIOSAMPLEDLG_H__26540673_FD9B_4799_889E_E05C3076016C__INCLUDED_)
