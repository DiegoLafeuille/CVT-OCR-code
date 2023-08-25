// GxSingleCamMonoDlg.h : header file
//

#if !defined(AFX_GxSingleCamMonoDLG_H__D8D963C2_3BE2_44EC_B0EB_FCE016DDD6FA__INCLUDED_)
#define AFX_GxSingleCamMonoDLG_H__D8D963C2_3BE2_44EC_B0EB_FCE016DDD6FA__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

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
// CGxSingleCamMonoDlg dialog
#include "GxIAPI.h"
#include "DxImageProc.h"
#include <vector>

class CGxSingleCamMonoDlg : public CDialog
{
// Construction
public:
	CGxSingleCamMonoDlg(CWnd* pParent = NULL);	// standard constructor
	
public:
	
	GX_STATUS InitDevice();

	
	GX_STATUS SetPixelFormat8bit();

	
	GX_STATUS GetDeviceParam();


	void InitUI();

	
	void InitUIParam(GX_DEV_HANDLE hDevice);

  
	void UpDateUI();
	
	
	void UnPrepareForShowImg();
	
	
	void DrawImg();
	
	
	void SaveImage();
	
	
	void ShowErrorString(GX_STATUS emErrorStatus);

	
	void InitEnumUI(GX_FEATURE_ID_CMD emFeatureID, CComboBox *pComboBox, bool bIsImplement);
	
	
	void InitGainUI();
	
	
	void InitShutterUI();

	
	void InitTriggerModeUI();

	
	void InitTriggerSourceUI();

	
	void InitTriggerActivationUI();


	static void __stdcall OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM* pFrame);

	
	bool PrepareForShowImg();
public:
	GX_DEV_HANDLE           m_hDevice;            ///< The handle for device  
	BITMAPINFO             *m_pBmpInfo;	          ///< The pointer for showing image 
	CString                 m_strFilePath;        ///< The path for saving image 
	int64_t                 m_nPayLoadSize;       ///< Image payload size
	int64_t                 m_nImageHeight;       ///< ImageHeight
    int64_t                 m_nImageWidth;        ///< ImageWidth
	int64_t                 m_nTriggerMode;       ///< TriggerMode
               
	GX_FLOAT_RANGE          m_stShutterFloatRange;///< The range of ExposureTime 
	GX_FLOAT_RANGE          m_stGainFloatRange;   ///< The range of Gain

	BYTE                   *m_pBufferRaw;        ///< The pointer for the raw data of image
	BYTE                   *m_pImageBuffer;      ///< The pointer for the image has been processed 
	char                    m_chBmpBuf[2048];    ///< The buffer for BMP image
	BOOL                    m_bIsSaveImg;        ///< The flag indicates whether the image is saved
	BOOL                    m_bDevOpened;        ///< The flag indicates whether the first device has been opened
	BOOL                    m_bIsSnap;           ///< The flag indicates whether image acquisition has started

	bool                    m_bTriggerMode;      ///< The flag indicates whether the device supports TriggerMode
	bool                    m_bTriggerActive;    ///< The flag indicates whether the device supports TriggerActivation
	bool                    m_bTriggerSource;    ///< The flag indicates whether the device supports TriggerSource

	CWnd                    *m_pWnd;             ///< The pointer for showing image window 

	HDC                     m_hDC;               ///< The handle for drawing image DC 

// Dialog Data
	//{{AFX_DATA(CGxSingleCamMonoDlg)
	enum { IDD = IDD_GxSingleCamMono_DIALOG };
	double	m_dShutterValue;
	double	m_dGainValue;
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CGxSingleCamMonoDlg)
	public:
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CGxSingleCamMonoDlg)
	virtual BOOL    OnInitDialog();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg void OnClose();
	afx_msg void OnBnClickedBtnSofttrigger();
	afx_msg void OnCbnSelchangeComboTriggerMode();
	afx_msg void OnBnClickedBtnOpenDevice();
	afx_msg void OnBnClickedBtnCloseDevice();
	afx_msg void OnBnClickedBtnStopSnap();
	afx_msg void OnBnClickedCheckSaveImage();
	afx_msg void OnBnClickedBtnStartSnap();
	afx_msg void OnCbnSelchangeComboTriggerActive();
	afx_msg void OnCbnSelchangeComboTriggerSource();
	afx_msg void OnKillfocusEditShutter();
	afx_msg void OnKillfocusEditGain();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_GxSingleCamMonoDLG_H__D8D963C2_3BE2_44EC_B0EB_FCE016DDD6FA__INCLUDED_)
