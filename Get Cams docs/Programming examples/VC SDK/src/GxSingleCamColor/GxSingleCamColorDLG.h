// GxSingleCamColorDlg.h : header file
//

#if !defined(AFX_GxSingleCamColorDLG_H__D8D963C2_3BE2_44EC_B0EB_FCE016DDD6FA__INCLUDED_)
#define AFX_GxSingleCamColorDLG_H__D8D963C2_3BE2_44EC_B0EB_FCE016DDD6FA__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#define  WHITE_BALANCE_RNTRY        3         
#define  DEFAULT_CONTRAST           0           
#define  DEFAULT_GAMMA              1         
#define  DEFAULT_COLOR_CORRECTION   0         

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
// CGxSingleCamColorDlg dialog
#include "GxIAPI.h"
#include "DxImageProc.h"


class CGxSingleCamColorDlg : public CDialog
{
// Construction
public:
	CGxSingleCamColorDlg(CWnd* pParent = NULL);	// standard constructor
	
public:
	GX_DEV_HANDLE       m_hDevice;              ///< The handle for device  

	BITMAPINFO          *m_pBmpInfo;	        ///< The pointer for showing image 
	BYTE                *m_pBufferRaw;          ///< The pointer for the raw data of image
	BYTE                *m_pBufferRGB;	        ///< The pointer for showing bmp image
    char                 m_chBmpBuf[2048];	    ///< The buffer for BMP image

	int64_t             m_nPayLoadSize;         ///< Image payload size
	int64_t             m_nImageHeight;         ///< ImageHeight
    int64_t             m_nImageWidth;          ///< ImageWidth
	int64_t             m_nPixelColorFilter;    ///< PixelColorFilter
	int64_t             m_nTriggerMode;         ///< TriggerMode
	int64_t             m_nAutoWhiteBalance;    ///< BalanceWhiteAuto
	int64_t             m_nRatioSelector;       ///< BalanceWhiteRatioSelector

	BYTE                *m_pGammaLut;           ///< The pointer for Gamma lookup table
	BYTE                *m_pContrastLut;        ///< The pointer for contrast lookup table
	int                 m_nLutLength;           ///< The length of the lookup table
	int64_t             m_nContrast;            ///< The value of contrast
	int64_t             m_nColorCorrection;     ///< The value of ColorCorrection
	double              m_dGamma;               ///< The value of Gamma

	GX_FLOAT_RANGE      m_stShutterFloatRange;  ///< The range of ExposureTime
	GX_FLOAT_RANGE      m_stGainFloatRange;     ///< The range of Gain
	GX_FLOAT_RANGE      m_stFloatRange;         ///< Save the range of BalanceRatio  

    CString             m_strFilePath;          ///< The path for saving image 
	BOOL                m_bIsSaveImg;           ///< The flag indicates whether the image is saved
	BOOL                m_bDevOpened;           ///< The flag indicates whether the first device has been opened
	BOOL                m_bIsSnap;              ///< The flag indicates whether image acquisition has started
	BOOL                m_bIsImproveImg;        ///< The flag indicates whether the image quality is improved
	bool                m_bTriggerMode;         ///< The flag indicates whether the device supports TriggerMode
	bool                m_bTriggerActive;       ///< The flag indicates whether the device supports TriggerActivation
	bool                m_bTriggerSource;       ///< The flag indicates whether the device supports TriggerSource 
	bool                m_bColorFilter;         ///< The flag indicates whether the device supports ColorFilter 
	bool                m_bBalanceRatioSelector;///< The flag indicates whether the device supports BalanceRatioSelector

	CWnd                *m_pWnd;                ///< The pointer for showing image window                 
	HDC                 m_hDC;                  ///< The handle for drawing image DC 

public:
	
	bool PrepareForShowImg();

	void InitUI();

   	void UpDateUI();
 
	static void __stdcall OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM* pFrame);
	
	void GetImageImproveParam();

	void UnPrepareForShowImg();

	void DrawImg();

	void SaveImage();

	void ShowErrorString(GX_STATUS emErrorStatus);

	void InitEnumUI(GX_FEATURE_ID_CMD emFeatureID, CComboBox *pComboBox, bool bIsImplement);

	void InitWhiteRatioUI();

	void RefreshWhiteRatio();

	void InitGainUI();

	void InitShutterUI();

	void InitTriggerModeUI();

	void InitTriggerSourceUI();

	void InitTriggerActivationUI();

	GX_STATUS InitDevice();

	GX_STATUS SetPixelFormat8bit();

	GX_STATUS GetDeviceParam();
// Dialog Data
	//{{AFX_DATA(CGxSingleCamColorDlg)
	enum { IDD = IDD_GXSINGLECAMCOLOR_DIALOG };
	double      m_dBalanceRatio;
	double	m_dGainValue;
	double	m_dShutterValue;
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CGxSingleCamColorDlg)
	public:
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CGxSingleCamColorDlg)
	virtual BOOL OnInitDialog();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg void OnClose();
	afx_msg void OnBnClickedBtnSofttrigger();
	afx_msg void OnCbnSelchangeComboTriggerMode();
	afx_msg void OnCbnSelchangeComboWriteBlance();
	afx_msg void OnCbnSelchangeComboTriggerSource();
	afx_msg void OnCbnSelchangeComboTriggerActive();
	afx_msg void OnBnClickedBtnOpenDevice();
	afx_msg void OnBnClickedBtnCloseDevice();
	afx_msg void OnBnClickedBtnStopSnap();
	afx_msg void OnBnClickedBtnStartSnap();
	afx_msg void OnBnClickedCheckSaveBmp();
	afx_msg void OnBnClickedCheckImageImprove();
	afx_msg void OnCbnSelchangeComboRatioSelector();
	afx_msg void OnEnKillfocusEditBalanceRatio();
	afx_msg void OnKillfocusEditShutter();
	afx_msg void OnKillfocusEditGain();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_GxSingleCamColorDLG_H__D8D963C2_3BE2_44EC_B0EB_FCE016DDD6FA__INCLUDED_)
