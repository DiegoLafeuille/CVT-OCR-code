// GxGetImageDlg.h : header file
//

#if !defined(AFX_GXGETIMAGEDLG_H__EA4AE0E8_97C9_47D9_B720_3F7E53EFC867__INCLUDED_)
#define AFX_GXGETIMAGEDLG_H__EA4AE0E8_97C9_47D9_B720_3F7E53EFC867__INCLUDED_

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

//----------------------------------------------------------------------------------
/**
\brief  This is a class for statistical time
*/
//----------------------------------------------------------------------------------
class CStatistics
{
public:

	CStatistics()
	{
		m_dSumTime = 0.0;  
		m_dMax     = 0.0;      
		m_dMin     = 0.0;    
		m_dAverage = 0.0; 
		m_nNum     = 0;      
	}

	void AddStatisticalData(double dData)
	{
		__IncSumTimeData(dData);
		__UpdateData(dData);
	}


	double GetMax()
	{
		return m_dMax;
	}

  
	double GetMin()
	{
		return m_dMin;
	}

	double GetAverage()
	{
		if (m_nNum == 0)
		{
			return 0;
		}

		return m_dSumTime / m_nNum;
	}

	void Reset()
	{
		m_dSumTime = 0.0;  
		m_dMax     = 0.0;      
		m_dMin     = 0.0;      
		m_dAverage = 0.0; 
		m_nNum     = 0;  
	}

private:
	void __IncSumTimeData(double dData)
	{
		m_dSumTime = m_dSumTime + dData;
		m_nNum++;
	}

	void __UpdateData(double dData)
	{
		if (m_nNum == 1)
		{
			m_dMax = dData;
			m_dMin = dData;
			return;
		}

		if (dData > m_dMax)
		{
			m_dMax = dData;
		}

		if (dData < m_dMin)
		{
			m_dMin = dData;
		}
	}

private:
	double  m_dSumTime;  
	double  m_dMax;     
	double  m_dMin;    
	double  m_dAverage;   
	int64_t m_nNum;     
};

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
		return (double)(timeDifference * 1000/ ticksPerSecond); 
	}
protected: 
	LARGE_INTEGER m_StartTime;   
    LARGE_INTEGER m_StopTime;    
};

/////////////////////////////////////////////////////////////////////////////
// CGxGetImageDlg dialog

class CGxGetImageDlg : public CDialog
{
// Construction
public:
	CGxGetImageDlg(CWnd* pParent = NULL);	// standard constructor

	GX_STATUS RegisterShowWnd(void* pWnd);

	GX_STATUS RegisterLogWnd(void* pWnd);

	GX_STATUS RegisterTimeWnd(void* pWnd);

	GX_STATUS OpenDevice();

	GX_STATUS CloseDevice();

	GX_STATUS InitDevice();

	GX_STATUS GetDeviceInitParam();

	GX_STATUS SetPixelFormat8bit();

	bool PrepareForShowColorImg();

	bool PrepareForShowMonoImg();

	bool PrepareForShowImg();
	
	void ShowImage(GX_FRAME_DATA* pFrameData);

	void DrawImg(void *pImageBuf, int64_t nImageWidth, int64_t nImageHeight);

	void UnPrepareForShowImg();

	void UpdateUI();

	void ShowErrorString(GX_STATUS emErrorStatus);

	void UpdateStatisticalData(double dData);
	
// Dialog Data
	//{{AFX_DATA(CGxGetImageDlg)
	enum { IDD = IDD_GXGETIMAGE_DIALOG };
	UINT	m_nEditTimeOut;
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CGxGetImageDlg)
	public:
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CGxGetImageDlg)
	virtual BOOL OnInitDialog();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg void OnClose();
	afx_msg void OnBnClickedBtnStartSnap();
	afx_msg void OnBnClickedBtnStopSnap();
	afx_msg void OnBnClickedBtnOpenDevice();
	afx_msg void OnBnClickedBtnCloseDevice();
	afx_msg void OnBnClickedBtnSendSoftCommand();
	afx_msg void OnBnClickedBtnCleanTime();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()

public:
	GX_DEV_HANDLE  m_hDevice;           ///< The handle for device          
	bool           m_bIsOpen;           ///< The flag indicates whether the first device has been opened
	bool           m_bIsSnap;           ///< The flag indicates whether AcquisitionStart command has been send 
	BITMAPINFO    *m_pBmpInfo;          ///< The pointer for showing image 
	char           m_chBmpBuf[2048];    ///< The buffer for showing image 
	int64_t        m_nImageWidth;       ///< ImageHeight
	int64_t        m_nImageHeight;      ///< ImageWidth
	int64_t        m_nPixelColorFilter; ///< PixelColorFilter
	int64_t        m_nPayLoadSize;      ///< Image payload size
	bool           m_bIsColorFilter;    ///< The flag indicates whether the acquired image is color filter
	BYTE          *m_pImgBuffer;        ///< The pointer for the image has been processed                
	GX_FRAME_DATA  m_stFrameData;       ///< The structure for image data                    


	CStatistics    m_objStatistic;      ///< Statistics
	CStopWatch     m_objStopTime;       ///< Stopwatch
   
	CWnd           *m_pWnd;             ///< The pointer for showing image window 
	HDC             m_hDC;              ///< The handle for drawing image DC 
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_GXGETIMAGEDLG_H__EA4AE0E8_97C9_47D9_B720_3F7E53EFC867__INCLUDED_)
