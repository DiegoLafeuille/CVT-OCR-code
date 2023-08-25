// GxCameraEventsDlg.h : header file
//

#if !defined(AFX_GxCameraEventsDLG_H__AA549272_EB77_49E3_8F5F_64128182B62A__INCLUDED_)
#define AFX_GxCameraEventsDLG_H__AA549272_EB77_49E3_8F5F_64128182B62A__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
#include <string>
#include <iostream>
using namespace std;

#include "GxIAPI.h"

/////////////////////////////////////////////////////////////////////////////
// CGxCameraEventsDlg dialog
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

//----------------------------------------------------------------------------------
/**
\brief  This is a class for stopwatch    
*/
//----------------------------------------------------------------------------------
class CStopWatch
{ 
public: 
	
	CStopWatch()
	{ 
		Start(); 
	} 
	
	void Start() 
	{ 
		QueryPerformanceCounter(&m_StartTime);
	} 
	
	double Stop() 
	{
		QueryPerformanceCounter(&m_StopTime);
		double theElapsedTime = ElapsedTime();
		
		m_StartTime = m_StopTime; 
		return theElapsedTime;
	} 
	
	double ElapsedTime() 
	{ 
		LARGE_INTEGER timerFrequency; QueryPerformanceFrequency(&timerFrequency); 
		__int64 oldTicks = ((__int64)m_StartTime.HighPart << 32) + (__int64)m_StartTime.LowPart;
		__int64 newTicks = ((__int64)m_StopTime.HighPart << 32) + (__int64)m_StopTime.LowPart; 
		long double timeDifference = (long double) (newTicks - oldTicks); 
		long double ticksPerSecond = (long double) (((__int64)timerFrequency.HighPart << 32) + (__int64)timerFrequency.LowPart); 
		return (double)(timeDifference * 1000 / ticksPerSecond); 
	}
protected: 
	LARGE_INTEGER m_StartTime;   
    LARGE_INTEGER m_StopTime;    
};

class CGxCameraEventsDlg : public CDialog
{
// Construction
public:
	CGxCameraEventsDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	//{{AFX_DATA(CGxCameraEventsDlg)
	enum { IDD = IDD_GxCameraEvents_DIALOG };
		// NOTE: the ClassWizard will add data members here
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CGxCameraEventsDlg)
	public:
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CGxCameraEventsDlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnClose();
	afx_msg void OnOpenDevice();
	afx_msg void OnStartSnap();
	afx_msg void OnStopSnap();
	afx_msg void OnCloseDevice();
	afx_msg void OnSendSoftCommand();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
public:

	GX_STATUS InitParam();

	static void __stdcall OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM *pFrame);

	static void __stdcall OnFeatureCallbackFun(GX_FEATURE_ID_CMD emFeatureID, void* pUserParam);
	
	void PrintInfotoWnd(string &strInfo);

	void ShowErrorString(GX_STATUS emErrorStatus);

	void UpdateUI();

private:
	GX_STATUS __IsSupportEvent(bool &bSupportExposureEndEvent);

private:
	GX_DEV_HANDLE              m_hDevice;                      ///< The handle for device  
	GX_FEATURE_CALLBACK_HANDLE m_hCB;                          ///< The handle for ExposureEndEvent callback function    
	void                      *m_pLogWnd;                      ///< The pointer for recording the time of the event
	bool                       m_bIsOpen;                      ///< The flag indicates whether the first device has been opened
	bool                       m_bIsSnap;                      ///< The flag indicates whether AcquisitionStart command has been send 
	CStopWatch                 m_objImageTime;                 ///< The stopwatch for receiving image
	CStopWatch                 m_objShutterTime;               ///< The stopwatch for completing the exposure event
	bool                       m_bImplementEventSelector;      ///< The flag indicates whether the device supports EventSourceSelector 
	bool                       m_bSupportExposureEndEvent;     ///< The flag indicates whether the device supports ExposureEndEvent
	bool                       m_bIsTrigValid;                 ///< The flag indicates whether the trigger is valid or not
};

/// This function demonstrates how to format log strings.
inline std::string  MsgComposer(const char* pFormat, ... )
{
	char    chBuffer[256] = {0};
	va_list  vap;
	va_start(vap, pFormat);
	
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

#endif // !defined(AFX_GxCameraEventsDLG_H__AA549272_EB77_49E3_8F5F_64128182B62A__INCLUDED_)
