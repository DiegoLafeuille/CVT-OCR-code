// GxTransferDelayDlg.h : header file
//

#if !defined(AFX_GxTransferDelayDLG_H__F071B8A4_DEDE_4E41_961A_11776A84A4F3__INCLUDED_)
#define AFX_GxTransferDelayDLG_H__F071B8A4_DEDE_4E41_961A_11776A84A4F3__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
#include "GxIAPI.h"
#include "DxImageProc.h"
#include "FileVersion.h"
#include <string>
#include <iostream>
using namespace std;

/////////////////////////////////////////////////////////////////////////////
// CGxTransferDelayDlg dialog
#define   TRANSFER_COUNT          1    
#define   MAX_NUM_CAMERA          4       
#define   CAMERA0                 0    
#define   CAMERA1                 1   
#define   CAMERA2                 2     
#define   CAMERA3                 3    
#define   CAM0_WAIT_TIME          2000  
#define   CAM1_WAIT_TIME          4000  
#define   CAM2_WAIT_TIME          6000  
#define   CAM3_WAIT_TIME          8000  

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

typedef struct CAMER_INFO
{
	BITMAPINFO					*pBmpInfo;		  ///< The pointer for showing image 
	BYTE						*pShowBuf;	      ///< The buffer for showing image      
	char						chBmpBuf[2048];	  ///< The buffer for BMP image
	int64_t					    nImageWidth;	  ///< ImageWidth
	int64_t					    nImageHeight;	  ///< ImageHeight
	int64_t					    nBayerLayout;	  ///< PixelColorFilter
	bool						bIsColorFilter;	  ///< Check whether the device supports ColorFilter
}CAMER_INFO;      

class CGxTransferDelayDlg : public CDialog
{
// Construction
public:
	CGxTransferDelayDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	//{{AFX_DATA(CGxTransferDelayDlg)
	enum { IDD = IDD_GxTransferDelay_DIALOG };
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CGxTransferDelayDlg)
	public:
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CGxTransferDelayDlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnClose();
	afx_msg void OnOpenDevice();
	afx_msg void OnStartSnap();
	afx_msg void OnStopSnap();
	afx_msg void OnCloseDevice();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
public:

	GX_STATUS SetCameraParam(GX_DEV_HANDLE hDevice);
	
	GX_STATUS SetPixelFormat8Bit(GX_DEV_HANDLE hDevice);

	GX_STATUS GetCameraParam(GX_DEV_HANDLE hDevice, CAMER_INFO *pstCamInfo);

	bool InitShowImageInfo(GX_DEV_HANDLE hDevice, CAMER_INFO *pstCamInfo);

	void UnInitShowImageInfo(BYTE **ppShowBuf);

	GX_STATUS RegisterCallback(int nDevNum);

	static void __stdcall OnFrameCallbackFun0(GX_FRAME_CALLBACK_PARAM *pFrame);

	static void __stdcall OnFrameCallbackFun1(GX_FRAME_CALLBACK_PARAM *pFrame);

	static void __stdcall OnFrameCallbackFun2(GX_FRAME_CALLBACK_PARAM *pFrame);

	static void __stdcall OnFrameCallbackFun3(GX_FRAME_CALLBACK_PARAM *pFrame);

    static void __stdcall OnFeatureCallbackFun0(GX_FEATURE_ID_CMD emFeatureID, void* pUserParam);

    static void __stdcall OnFeatureCallbackFun1(GX_FEATURE_ID_CMD emFeatureID, void* pUserParam);

    static void __stdcall OnFeatureCallbackFun2(GX_FEATURE_ID_CMD emFeatureID, void* pUserParam);

    static void __stdcall OnFeatureCallbackFun3(GX_FEATURE_ID_CMD emFeatureID, void* pUserParam);

	void DrawImage(BYTE *pImageBuf, int nWidth, int nHeight, BITMAPINFO *pBmpInfo, HDC hDC, RECT objRect);

	void ShowErrorString(GX_STATUS emErrorStatus);

	bool InitResources();

	void ReleaseResource();

	void UpdateUI();

	void PrintInfotoWnd(string &strInfo);

private:
	int                         m_nDevNum;              ///< The number of connected devices 
	GX_DEV_HANDLE*	            m_pDeviceHandle;	    ///< The pointer of handle
	GX_FEATURE_CALLBACK_HANDLE* m_pCBHandle;            ///< The pointer of the exposure callback function
	CAMER_INFO*	  	            m_pstCamInfo;		    ///< The pointer of the camera data structure 
	bool                        m_bIsOpen;              ///< The flag indicates whether the device has been opened
	bool                        m_bIsSnap;              ///< The flag indicates whether image acquisition has started
	HANDLE                      m_hCam0StartTranEvent;  ///< The handle for Camera No.0 StartTransferEvent 
	HANDLE                      m_hCam1StartTranEvent;  ///< The handle for Camera No.1 StartTransferEvent 
	HANDLE                      m_hCam2StartTranEvent;  ///< The handle for Camera No.2 StartTransferEvent  
	HANDLE                      m_hCam3StartTranEvent;  ///< The handle for Camera No.3 StartTransferEvent 
	HANDLE                      m_hCam0StopSnapEvent;   ///< The handle for Camera No.0 AcquisitionStopEvent 
	HANDLE                      m_hCam1StopSnapEvent;   ///< The handle for Camera No.1 AcquisitionStopEvent 
	HANDLE                      m_hCam2StopSnapEvent;   ///< The handle for Camera No.2 AcquisitionStopEvent 
	HANDLE                      m_hCam3StopSnapEvent;   ///< The handle for Camera No.3 AcquisitionStopEvent 
	void                        *m_pLogWnd;             ///< The pointer of the log window 
	HWND                        m_hWndHandle[4];        ///< The window handle for showing image 
    HDC                         m_hDC[4];               ///< The handle for drawing image DC 
	RECT                        m_objRect[4];           ///< The size of the image window 

	
};

inline std::string  MsgComposer( const char* pFormat, ... )
{
	char pBuffer[256] = {0};
	va_list vap;
	va_start( vap, pFormat );
	
#if _MSC_VER >= 1400 
	_vsnprintf_s( pBuffer, 256, pFormat, vap );
#else if _MSC_VER == 1200 
	_vsnprintf( pBuffer, 256, pFormat, vap );
#endif
	
	va_end(vap);
	return std::string( pBuffer);
	
};


//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_GxTransferDelayDLG_H__F071B8A4_DEDE_4E41_961A_11776A84A4F3__INCLUDED_)
