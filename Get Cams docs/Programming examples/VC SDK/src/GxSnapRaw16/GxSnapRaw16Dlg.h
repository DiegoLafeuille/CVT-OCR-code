// GxSnapRaw16Dlg.h : header file
//

#if !defined(AFX_GXSNAPRAW16DLG_H__DE201059_C44D_42D2_A515_AE83A37BF98D__INCLUDED_)
#define AFX_GXSNAPRAW16DLG_H__DE201059_C44D_42D2_A515_AE83A37BF98D__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

/////////////////////////////////////////////////////////////////////////////
// CGxSnapRaw16Dlg dialog

#include "GxIAPI.h"
#include "DxImageProc.h"

#define  GX_VERIFY(emStatus) \
	                            if(emStatus != GX_STATUS_SUCCESS) \
								{\
								   ShowErrorString(emStatus); \
								   return;\
								} 

#define VERIFY_STATUS_RET(emStatus) \
	                               if (emStatus != GX_STATUS_SUCCESS) \
                                   {\
								      return emStatus;\
                                   }\

class CGxSnapRaw16Dlg : public CDialog
{
// Construction
public:
	CGxSnapRaw16Dlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	//{{AFX_DATA(CGxSnapRaw16Dlg)
	enum { IDD = IDD_GXSNAPRAW16_DIALOG };
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CGxSnapRaw16Dlg)
	public:
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CGxSnapRaw16Dlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnOpenDevice();
	afx_msg void OnStartSnap();
	afx_msg void OnStopSnap();
	afx_msg void OnCloseDevice();
	afx_msg void OnClose();
	afx_msg void OnSaveRawImage();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
public:
	
	
	GX_STATUS InitParam();

	GX_STATUS SetPixelSize16(CString *pstrSymbolic);

	bool PrepareForShowImage();
	
	void UnPrepareForShowImage();
	
	static void __stdcall OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM *pFrame);
	
	void DrawImage(BYTE *pImageBuf, int nWidth, int nHeight, BITMAPINFO *pBmpInfo);

	void SaveRawImage(BYTE *pRawBuf, int nImageSize, CString srtSavePath);
	
	void ShowErrorString(GX_STATUS emErrorStatus);

	void UpdateUI();

private: 
	CString                 m_strSavePath;            ///< The path for saving image 
	GX_DEV_HANDLE           m_hDevice;                ///< The handle for device  
	BITMAPINFO             *m_pBmpInfo;	              ///< The pointer for showing BMP image 
	BYTE                   *m_pRawBuf;                ///< The pointer of the raw image buffer
	BYTE                   *m_pRaw8Buf;               ///< The pointer of 8Bit raw image buffer                 
	BYTE                   *m_pShowBuf;               ///< The pointer for showing image 
	char					m_chBmpBuf[2048];	      ///< The buffer for showing image
	int64_t                 m_nImageWidth;            ///< Acquired image width
	int64_t                 m_nImageHeight;           ///< Acquired image height
	int64_t                 m_nPayLoadSize;           ///< Acquired raw image size 
	int64_t                 m_nPixelColorFilter;      ///< PixelColorFilter 
	bool                    m_bIsColorFilter;         ///< The flag indicates whether the device supports ColorFilter
	bool                    m_bIsOpen;                ///< The flag indicates whether the first device has been opened
	bool                    m_bIsSnap;                ///< The flag indicates whether AcquisitionStart command has been send 
	bool                    m_bIsSaveImg;             ///< The flag indicates whether the image is saved
	HWND                    m_hWndHandle;             ///< The handle for showing image window     
    HDC                     m_hDC;                    ///< The handle for drawing image DC 
	RECT                    m_objRect;                ///< The size of image window

};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_GXSNAPRAW16DLG_H__DE201059_C44D_42D2_A515_AE83A37BF98D__INCLUDED_)
