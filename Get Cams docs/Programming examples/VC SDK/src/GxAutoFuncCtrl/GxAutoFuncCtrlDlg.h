// GxAutoFuncCtrlDlg.h : header file
//

#if !defined(AFX_GXAUTOFUNCCTRLDLG_H__48794C28_712D_4E1F_97EA_58B5A231704F__INCLUDED_)
#define AFX_GXAUTOFUNCCTRLDLG_H__48794C28_712D_4E1F_97EA_58B5A231704F__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "GxIAPI.h"
#include "DxImageProc.h"
#include <string>
#include <iostream>
using namespace std;

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

/////////////////////////////////////////////////////////////////////////////
// CGxAutoFuncCtrlDlg dialog

class CGxAutoFuncCtrlDlg : public CDialog
{
// Construction
public:
	CGxAutoFuncCtrlDlg(CWnd* pParent = NULL);	// standard constructor
public:
 
	static void __stdcall OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM *pFrame);

	GX_STATUS GetDeviceParam();

	GX_STATUS SetPixelFormat8bit();

	GX_STATUS InitDevice();

	bool PrepareForShowColorImg();

	bool PrepareForShowMonoImg();

	bool PrepareForShowImg();

	void UpdateROIRange();

	void UnPrepareForShowImg();

	void InitEnumUI(GX_FEATURE_ID_CMD emFeatureID, CComboBox *pComboBox, bool bIsImplemented);

	void DrawImg(BYTE *pImageBuffer);

	void InitUI();

	void InitShutterUI();

	void InitGainUI();

	void InitGrayUI();

	void InitAutoGainUI();

	void InitROIUI();

	void InitAutoShutterUI();

	void InitLightEnvironmentUI();
	
	void UpDateUI();

	void RefreshCurValue();

	void ShowErrorString(GX_STATUS emErrorStatus);

// Dialog Data
	//{{AFX_DATA(CGxAutoFuncCtrlDlg)
	enum { IDD = IDD_GXAUTOFUNCCTRL_DIALOG };
	int		m_nGray;               ///< ExpectedGrayValue
	int		m_nRoiX;               ///< AA ROIOffsetX
	int		m_nRoiY;               ///< AA ROIOffsetY
	int		m_nRoiH;               ///< AA ROIHeight
	int		m_nRoiW;               ///< AA ROIWidth
	double	m_dAutoGainMin;               
	double  m_dAutoGainMax;      
	double	m_dAutoShutterMin;    
	double	m_dAutoShutterMax;    
	double	m_dShutterValue;     
	double	m_dGainValue;          
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CGxAutoFuncCtrlDlg)
	public:
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CGxAutoFuncCtrlDlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnBnClickedBtnOpenDevice();
	afx_msg void OnBnClickedBtnStartSnap();
	afx_msg void OnBnClickedBtnStopSnap();
	afx_msg void OnBnClickedBtnCloseDevice();
	afx_msg void OnCbnSelchangeComboLight();
	afx_msg void OnCbnSelchangeComboAutoGain();
	afx_msg void OnCbnSelchangeComboAutoShutter();
	afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
	afx_msg void OnKillfocusEditRoiX();
	afx_msg void OnKillfocusEditRioY();
	afx_msg void OnKillfocusEditRoiW();
	afx_msg void OnKillfocusEditRoiH();
	afx_msg void OnKillfocusEditAutoGainMax();
	afx_msg void OnKillfocusEditAutoGainMin();
	afx_msg void OnKillfocusEditAutoShutterMax();
	afx_msg void OnKillfocusEditAutoShutterMin();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnClose();
	afx_msg void OnKillfocusEditCurShutter();
	afx_msg void OnKillfocusEditCurGain();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()    

public:
	GX_DEV_HANDLE           m_hDevice;                ///< The handle of device 
	int64_t                 m_nImageWidth;            ///< Acquired ImageWidth
	int64_t                 m_nImageHeight;           ///< Acquired ImageHeight
	int64_t                 m_nPayLoadSize;           ///< Image payload size
	int64_t                 m_nPixelColorFilter;      ///< PixelColorFilter

	GX_FLOAT_RANGE          m_stShutterFloatRange;    ///< The range of ExposureTime
	GX_FLOAT_RANGE          m_stGainFloatRange;       ///< The range of Gain

	BITMAPINFO             *m_pBmpInfo;	              ///< The pointer for showing image 

	BOOL                    m_bDevOpened;             ///< The flag indicates whether the first device has been opened
	BOOL                    m_bIsSnap;                ///< The flag indicates whether image acquisition has started
	bool                    m_bIsColorFilter;         ///< The flag indicates whether the device supports ColorFilter
	bool                    m_bImplementAutoGain;     ///< The flag indicates whether the device supports AutoGain
	bool                    m_bImplementAutoShutter;  ///< The flag indicates whether the device supports AutoExposure
	bool                    m_bImplementLight;        ///< The flag indicates whether the device supports AA LightEnvironment

	BYTE                   *m_pRawBuffer;             ///< The pointer for the raw data of image
	BYTE                   *m_pImgBuffer;             ///< The pointer for the image has been processed   
	char                    m_chBmpBuf[2048];         ///< The buffer for showing image 
	GX_EXPOSURE_AUTO_ENTRY  m_emAutoShutterMode;      ///< AutoExposureMode
	GX_GAIN_AUTO_ENTRY      m_emAutoGainMode;         ///< AutoGainMode

	CWnd                   *m_pWnd;                   ///< The pointer for showing image window 
	HDC                     m_hDC;                    ///< The handle for drawing image DC
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_GXAUTOFUNCCTRLDLG_H__48794C28_712D_4E1F_97EA_58B5A231704F__INCLUDED_)
