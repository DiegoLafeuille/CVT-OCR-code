                                                                                                                                                                                                                                                                                                                                                      // GxStoreAVIDlg.h : header file
//

#if !defined(AFX_GXSTOREAVIDLG_H__A8E25217_6814_45C7_AF97_010FAB1249DC__INCLUDED_)
#define AFX_GXSTOREAVIDLG_H__A8E25217_6814_45C7_AF97_010FAB1249DC__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "GxIAPI.h"
#include "DxImageProc.h"
#include <string>
#include <iostream>
using namespace std;

#include <wtypes.h>
#include <atlbase.h>
#include <shlobj.h>

#define _WIN32_DCOM
#include <ObjBase.h>
#pragma comment(lib, "ole32.lib")

#include "Vfw.h"
#pragma comment(lib, "Vfw32.lib")

#define  AVI_MAX_SIZE   0x80000000  ///< The max size of the AVI file is 2GB

#define AVI_FRAME_OFFSET  8

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

class CVxCriticalSection  
{
public:

	CVxCriticalSection () {InitializeCriticalSection(&m_cs);}
	
	virtual ~CVxCriticalSection() {DeleteCriticalSection(&m_cs);}
public:
	
	void Enter(){EnterCriticalSection(&m_cs);}

	void Leave(){LeaveCriticalSection(&m_cs);}
private:
	CRITICAL_SECTION m_cs; 
};


/////////////////////////////////////////////////////////////////////////////
// CGxStoreAVIDlg dialog
class CGxStoreAVIDlg : public CDialog
{
// Construction
public:
	CGxStoreAVIDlg(CWnd* pParent = NULL);	// standard constructor

	static void __stdcall OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM *pFrame);

	void UpDateUI();

	void ShowErrorString(GX_STATUS emErrorStatus);

	GX_STATUS GetDeviceParam();

	GX_STATUS InitDevice();

	GX_STATUS SetPixelFormat8bit();

	bool PrepareForShowColorImg();

	bool PrepareForShowMonoImg();

	bool PrepareForShowImg();

	void UnPrepareForShowImg();

	void DrawImg(BYTE *pImageBuffer);

	BOOL PrepareForSaveAVI();

	void UnPrepareForSaveAVI();

	void ExSaveAVI(BYTE * pImageBuf);

	void SaveAVI(BYTE *pImageBuf);

	int64_t GetDiskFreeSpace(); 

	void ResetCompressor();

	BOOL ReCreatVedioFile();

public:
	bool       m_bEnableSaveAVI;              ///< The flag indicates whether the program is saving AVI
	bool       m_bIsColorFilter;              ///< The flag indicates whether the device supports ColorFilter
	BOOL       m_bDevOpened;                  ///< The flag indicates whether the first device has been opened
	BYTE       *m_pRawBuffer;                 ///< The pointer for the raw data of image
	BYTE       *m_pImgBuffer;                 ///< The pointer for the image has been processed  
	int64_t    m_nPayLoadSize;                ///< Image payload size
	int64_t    m_nImageWidth;                 ///< ImageWidth
	int64_t    m_nImageHeight;                ///< ImageHeight
    int64_t    m_nPixelColorFilter;           ///< PixelColorFilter
	int64_t    m_nFrameSize;                  ///< The FrameSize of the AVI file
	int64_t    m_nWrittenDataSize;            ///< The DataSize of the AVI file
	char       m_chBmpBuf[2048];              ///< The buffer for showing image 

	BITMAPINFO    *m_pBmpInfo;	              ///< The pointer for showing image
	GX_DEV_HANDLE  m_hDevice;                 ///< The handle for device 

	PAVIFILE       m_pAVIFile;                ///< The handle for AVI file
	PAVISTREAM     m_ps;                      ///< The stream of AVI file
	PAVISTREAM	   m_psCompressed;            ///< The compressed stream
	LONG           m_nTimeFrame;              ///< The frame ID for writing AVI file
	HIC            m_hic;                     ///< The handle for compress video                    
	COMPVARS       m_Com;                     ///< The parameters for selecting the compressor
    CVxCriticalSection m_SaverCriSec;         ///< The mutex for saving
 
	CWnd                    *m_pWnd;          ///< The pointer for showing image window 
	HDC                     m_hDC;            ///< The handle for drawing image DC 

// Dialog Data
	//{{AFX_DATA(CGxStoreAVIDlg)
	enum { IDD = IDD_GXSTOREAVI_DIALOG };
	CString	m_strAVIFolder;
	BOOL	m_bCompressedAVI;
	BOOL	m_bShowImage;
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CGxStoreAVIDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CGxStoreAVIDlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnBnClickedBtnOpenDevice();
	afx_msg void OnBnClickedBtnCloseDevice();
	afx_msg void OnBnClickedBtnSaveAs();
	afx_msg void OnBnClickedBtnAviStart();
	afx_msg void OnBnClickedBtnStopAvi();
	afx_msg void OnBnClickedCheckCompressedAvi();
	afx_msg void OnBnClickedCheckShowImage();
	afx_msg void OnClose();
	afx_msg void OnBnClickedBtnSelectCompressor();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()

public:
	DWORD m_wPlayFPS;
	afx_msg void OnEnKillfocusEditFps();
	virtual BOOL PreTranslateMessage(MSG* pMsg);
};

inline std::string  MsgComposer( const char* pFormat, ... )
{
	char chBuffer[256] = {0};
	va_list vap;
	va_start( vap, pFormat );

#if _MSC_VER >= 1400 
	_vsnprintf_s(chBuffer, 256, pFormat, vap);
#else if _MSC_VER == 1200 
	_vsnprintf(chBuffer, 256, pFormat, vap );
#endif

	va_end(vap);
	return std::string(chBuffer);
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_GXSTOREAVIDLG_H__A8E25217_6814_45C7_AF97_010FAB1249DC__INCLUDED_)
