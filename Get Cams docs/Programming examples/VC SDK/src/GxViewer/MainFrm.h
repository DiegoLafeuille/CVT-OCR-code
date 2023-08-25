// MainFrm.h : interface of the CMainFrame class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_MAINFRM_H__561D6510_C385_4CB3_A6EA_80DA13D81C5E__INCLUDED_)
#define AFX_MAINFRM_H__561D6510_C385_4CB3_A6EA_80DA13D81C5E__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

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

class CMainFrame : public CFrameWnd
{
	
protected: // create from serialization only
	CMainFrame();
	DECLARE_DYNCREATE(CMainFrame)

// Attributes
public:

// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CMainFrame)
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
	//}}AFX_VIRTUAL

// Implementation
public:
	virtual ~CMainFrame();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

// Generated message map functions
protected:
	//{{AFX_MSG(CMainFrame)
	afx_msg void OnOpenDevice();
	afx_msg void OnStartSnap();
	afx_msg void OnStopSnap();
	afx_msg void OnCloseDevice();
	afx_msg void OnClose();
	afx_msg void OnUpdateOpenDevice(CCmdUI* pCmdUI);
	afx_msg void OnUpdateStartSnap(CCmdUI* pCmdUI);
	afx_msg void OnUpdateStopSnap(CCmdUI* pCmdUI);
	afx_msg void OnUpdateCloseDevice(CCmdUI* pCmdUI);
	afx_msg void OnPaint();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
public:

	
	GX_STATUS InitParam();

	
	GX_STATUS SetPixelFormat8Bit();

	
	bool PrepareForShowImage();

	
	void UnPrepareForShowImage();


	static void __stdcall OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM *pFrame);

	
	void DrawImage(BYTE *pImageBuf, int nWidth, int nHeight, BITMAPINFO *pBmpInfo);

	
	void ShowErrorString(GX_STATUS emErrorStatus);

private:
	GX_DEV_HANDLE           m_hDevice;                ///< The handle of device 
	BITMAPINFO             *m_pBmpInfo;	              ///< The pointer for showing image 
	BYTE                   *m_pRawBuf;                ///< The pointer for showing raw image 
	BYTE                   *m_pShowBuf;               ///< The pointer for showing buffer
	char					m_chBmpBuf[2048];	      ///< The buffer for showing image
	int64_t                 m_nImageWidth;            ///< Acquired ImageWidth
	int64_t                 m_nImageHeight;           ///< Acquired ImageHeight
	int64_t                 m_nPayLoadSize;           ///< Acquired raw image size
	int64_t                 m_nPixelColorFilter;      ///< PixelColorFilter 
	bool                    m_bIsColorFilter;         ///< Check whether the device supports PixelColorFilter
	bool                    m_bIsOpen;                ///< The flag indicates whether the first device has been opened
	bool                    m_bIsSnap;                ///< The flag indicates whether AcquisitionStart command has been send 

	HDC                     m_hDC;                    ///< The handle for drawing image DC 
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_MAINFRM_H__561D6510_C385_4CB3_A6EA_80DA13D81C5E__INCLUDED_)
